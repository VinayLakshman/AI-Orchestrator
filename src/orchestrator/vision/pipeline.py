from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..settings import Settings
from .detector import collect_latest_message_images, extract_latest_user_text, infer_vision_task, strip_images_from_messages
from .fetcher import resolve_image_ref
from .models import ResolvedImage, VisionAnalysis, VisionResult, VisionTaskType
from .prompts import build_vision_system_prompt, render_vision_context

logger = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client
        self._cache: dict[str, VisionResult] = {}

    @staticmethod
    def _parse_ollama_content(response: Any) -> str:
        try:
            if isinstance(response, dict):
                if "message" in response and isinstance(response["message"], dict):
                    return response["message"].get("content", "") or ""

                if "response" in response:
                    return str(response.get("response") or "").strip()

                if "content" in response:
                    content = response["content"]
                    if isinstance(content, list):
                        return "\n".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        ).strip()
                    return str(content or "").strip()

                if "choices" in response and response["choices"]:
                    return response["choices"][0].get("message", {}).get("content", "") or ""

            return str(response or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return {}

        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        start = text.find("{")
        end = text.rfind("}")
        candidate = text

        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]

        return json.loads(candidate)

    def _build_fallback_analysis(
        self,
        *,
        task_type: VisionTaskType,
        image_count: int,
        hashes: list[str],
        raw_text: str,
    ) -> VisionAnalysis:
        fallback = (raw_text or "").strip()
        if not fallback:
            fallback = "Vision model returned no structured output."

        return VisionAnalysis(
            task_type=task_type,
            confidence=0.35,
            summary=fallback[:500],
            ocr="",
            layout="",
            metrics="",
            errors_warnings="",
            observations=fallback[:1000],
            answer_context=fallback[:1500],
            image_count=image_count,
            source_model=self.settings.vision_model,
            raw_text=raw_text,
            hashes=hashes,
        )

    def _parse_analysis(
        self,
        *,
        task_type: VisionTaskType,
        image_count: int,
        hashes: list[str],
        raw_text: str,
    ) -> VisionAnalysis:
        try:
            parsed = self._extract_json_object(raw_text)
        except Exception:
            return self._build_fallback_analysis(
                task_type=task_type,
                image_count=image_count,
                hashes=hashes,
                raw_text=raw_text,
            )

        if not parsed:
            return self._build_fallback_analysis(
                task_type=task_type,
                image_count=image_count,
                hashes=hashes,
                raw_text=raw_text,
            )

        normalized = {
            "task_type": str(parsed.get("task_type") or task_type.value).strip().lower(),
            "confidence": float(parsed.get("confidence") or 0.7),
            "summary": str(parsed.get("summary") or "").strip(),
            "ocr": str(parsed.get("ocr") or "").strip(),
            "layout": str(parsed.get("layout") or "").strip(),
            "metrics": str(parsed.get("metrics") or "").strip(),
            "errors_warnings": str(parsed.get("errors_warnings") or parsed.get("errors") or "").strip(),
            "observations": str(parsed.get("observations") or "").strip(),
            "answer_context": str(parsed.get("answer_context") or "").strip(),
            "image_count": image_count,
            "source_model": self.settings.vision_model,
            "raw_text": raw_text,
            "hashes": hashes,
        }

        if not normalized["summary"]:
            normalized["summary"] = normalized["answer_context"] or normalized["observations"] or raw_text[:500]

        if not normalized["answer_context"]:
            normalized["answer_context"] = normalized["summary"]

        try:
            return VisionAnalysis.model_validate(normalized)
        except Exception:
            return self._build_fallback_analysis(
                task_type=task_type,
                image_count=image_count,
                hashes=hashes,
                raw_text=raw_text,
            )

    async def process(self, state: dict[str, Any]) -> VisionResult | None:
        messages = state.get("messages", []) or []
        metadata = state.get("metadata", {}) or {}
        request_headers = metadata.get("request_headers", {}) or {}

        images = collect_latest_message_images(messages, self.settings.vision_max_images)
        cleaned_messages = strip_images_from_messages(messages)

        if not images:
            return None

        user_text = extract_latest_user_text(messages)
        task_type = infer_vision_task(user_text)

        resolved_images: list[ResolvedImage] = []
        for ref in images:
            resolved = await resolve_image_ref(
                ref,
                settings=self.settings,
                headers=request_headers,
                client=self.client,
            )
            if resolved is not None:
                resolved_images.append(resolved)

        if not resolved_images:
            return None

        image_hashes = [img.sha256 for img in resolved_images]
        cache_key = "|".join([task_type.value, user_text.strip()[:512], *image_hashes])

        if cache_key in self._cache:
            cached = self._cache[cache_key].model_copy(deep=True)
            cached.cleaned_messages = cleaned_messages
            cached.cache_hit = True
            return cached

        system_prompt = build_vision_system_prompt(
            task_type=task_type,
            image_count=len(resolved_images),
            user_text=user_text,
        )

        user_prompt = (
            user_text.strip()
            if user_text.strip()
            else "Analyse the attached image(s) and return structured technical context."
        )

        payload = {
            "model": self.settings.vision_model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [img.base64_data for img in resolved_images],
                },
            ],
            "options": {
                "temperature": 0.15,
                "num_predict": 1200,
            },
        }

        try:
            resp = await self.client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_content = self._parse_ollama_content(data)

            analysis = self._parse_analysis(
                task_type=task_type,
                image_count=len(resolved_images),
                hashes=image_hashes,
                raw_text=raw_content,
            )

            context_markdown = render_vision_context(analysis)

            result = VisionResult(
                analysis=analysis,
                context_markdown=context_markdown,
                cleaned_messages=cleaned_messages,
                image_hashes=image_hashes,
                cache_hit=False,
            )

            self._cache[cache_key] = result.model_copy(deep=True)
            return result

        except Exception as exc:
            logger.exception("Vision pipeline failed: %s", exc)
            return VisionResult(
                analysis=self._build_fallback_analysis(
                    task_type=task_type,
                    image_count=len(resolved_images),
                    hashes=image_hashes,
                    raw_text=str(exc),
                ),
                context_markdown="",
                cleaned_messages=cleaned_messages,
                image_hashes=image_hashes,
                cache_hit=False,
            )