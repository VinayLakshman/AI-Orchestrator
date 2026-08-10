from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..logging import get_logger
from ..common.enums import VisionTaskType
from ..context.assembler import build_conversation
from ..settings import Settings
from .detector import infer_vision_task
from .fetcher import (
    resolve_image_ref,
    strip_images_from_messages,
)
from ..models.chat import ChatMessage
from ..models.vision import ResolvedImage, VisionAnalysis, VisionResult
from ..models.ollama import extract_assistant_text
from ..models.state import OrchestratorState
from .prompts import build_vision_system_prompt, render_vision_context

logger = get_logger(__name__)


@dataclass
class VisionPipeline:
    settings: Settings
    client: httpx.AsyncClient
    model_client: Any
    _cache: dict[str, VisionResult] = field(default_factory=dict)

    def _image_source_kind(self, ref: str) -> str:
        if ref.startswith("data:image/"):
            return "data_uri"
        if ref.startswith(("http://", "https://")):
            return "http_url"
        if ref.startswith("<"):
            return "placeholder"
        return "other"

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end != -1 and end > start else text
        try:
            import json
            return json.loads(candidate)
        except Exception:
            return {}

    def _build_no_resolved_images_result(
        self,
        *,
        task_type: VisionTaskType,
        cleaned_messages: list[dict[str, Any]],
        reason: str,
    ) -> VisionResult:
        summary = "No usable image payload was available to the vision pipeline."
        analysis = VisionAnalysis(
            task_type=task_type,
            confidence=0.0,
            summary=summary,
            ocr="",
            layout="",
            metrics="",
            errors_warnings="",
            observations=summary,
            answer_context=summary,
            image_count=0,
            source_model=self.settings.vision_model_name,
            raw_text=reason,
            hashes=[],
        )
        return VisionResult(
            analysis=analysis,
            context_markdown=render_vision_context(analysis),
            cleaned_messages=cleaned_messages,
            image_hashes=[],
            cache_hit=False,
        )

    def _build_fallback_analysis(
        self,
        *,
        task_type: VisionTaskType,
        image_count: int,
        hashes: list[str],
        raw_text: str,
    ) -> VisionAnalysis:
        summary = raw_text.strip()[:500] or f"Analyzed {image_count} image(s)."
        return VisionAnalysis(
            task_type=task_type,
            confidence=0.55,
            summary=summary,
            ocr="",
            layout="",
            metrics="",
            errors_warnings="",
            observations=summary,
            answer_context=summary,
            image_count=image_count,
            source_model=self.settings.vision_model_name,
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
            "source_model": self.settings.vision_model_name,
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

    async def process(self, state: OrchestratorState) -> VisionResult | None:
        messages = state.request.messages
        request_headers = state.request.metadata.get("request_headers", {}) or {}
        request_id = str(state.request.request_id or "")

        images = state.request.images[: self.settings.vision_max_images]
        cleaned_messages = strip_images_from_messages(messages)
        request_image_sources = [self._image_source_kind(ref) for ref in images]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "VISION REQUEST request_id=%s model=%s request_state_image_count=%d request_state_image_source=%s prompt_length=%d",
                request_id,
                self.settings.vision_model_name,
                len(images),
                request_image_sources,
                len(state.request.user_message or ""),
            )

        if not images:
            return None

        user_text = state.request.user_message
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "VISION REQUEST SKIPPED request_id=%s model=%s vision_skipped=true reason=%s resolved_image_count=%d",
                    request_id,
                    self.settings.vision_model_name,
                    "no_resolved_images",
                    0,
                )
            return self._build_no_resolved_images_result(
                task_type=task_type,
                cleaned_messages=cleaned_messages,
                reason="no_resolved_images",
            )

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
        if logger.isEnabledFor(logging.DEBUG):
            payload_encoded_lengths = [len(img.base64_data or "") for img in resolved_images]
            payload_raw_sizes = [len(base64.b64decode(img.base64_data)) for img in resolved_images]
            payload_mime_types = [img.mime_type for img in resolved_images]

            logger.debug(
                "VISION PAYLOAD CREATED request_id=%s model=%s payload_image_count=%d payload_image_source=%s payload_mime=%s payload_raw_size=%s payload_encoded_length=%s prompt_length=%d",
                request_id,
                self.settings.vision_model_name,
                len(resolved_images),
                request_image_sources,
                payload_mime_types,
                payload_raw_sizes,
                payload_encoded_lengths,
                len(user_prompt),
            )

        # Build OpenAI-compatible multimodal messages via the centralized
        # assembler. Exactly one SYSTEM message is emitted and the latest user
        # request (including image parts) becomes a real USER message.
        user_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for img in resolved_images:
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.mime_type};base64,{img.base64_data}"},
                }
            )
        chat_messages: list[ChatMessage] = build_conversation(
            system_prompt=system_prompt,
            latest_user_message=user_parts,
        )

        try:
            # Use the generic OpenAI-compatible llama.cpp client for inference.
            response = await self.model_client.chat(
                model=self.settings.vision_model_name,
                messages=chat_messages,
                temperature=0.15,
                max_tokens=1200,
                stream=False,
            )
            raw_content = extract_assistant_text(response.content)

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
                context_markdown=render_vision_context(
                    self._build_fallback_analysis(
                        task_type=task_type,
                        image_count=len(resolved_images),
                        hashes=image_hashes,
                        raw_text=str(exc),
                    )
                ),
                cleaned_messages=cleaned_messages,
                image_hashes=image_hashes,
                cache_hit=False,
            )
