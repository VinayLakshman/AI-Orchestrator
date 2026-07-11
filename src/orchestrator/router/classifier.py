from __future__ import annotations

import json
from typing import Any

import httpx

from ..schemas import ChatMessage, RouteDecision, RouteType
from ..settings import Settings


class RoutingClassifier:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    def _build_prompt(self, user_text: str) -> list[dict[str, str]]:
        system = (
            "You are a routing classifier for an AI orchestrator. "
            "Choose exactly one route from: general, vision, code, rag, tools, multi_step, clarify. "
            "Return only strict JSON with keys: route, confidence, reason, needs_vision, needs_rag, needs_tools, needs_code, needs_planning."
        )
        user = f"Classify this request:\n\n{user_text}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def classify(self, user_text: str) -> RouteDecision:
        if not self.settings.router_model:
            return RouteDecision(
                route=RouteType.GENERAL,
                confidence=0.5,
                reason="No router model configured; defaulting to general route.",
            )

        payload = {
            "model": self.settings.router_model,
            "messages": self._build_prompt(user_text),
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 128,
            },
        }

        timeout = httpx.Timeout(self.settings.router_timeout_s)
        close_client = False
        client = self.client
        if client is None:
            client = httpx.AsyncClient(base_url=self.settings.ollama_base_url, timeout=timeout)
            close_client = True

        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            parsed = self._parse_json(content)
            return RouteDecision(**parsed)
        except Exception:
            return RouteDecision(
                route=RouteType.GENERAL,
                confidence=0.35,
                reason="Router model failed; falling back to general route.",
            )
        finally:
            if close_client:
                await client.aclose()

    def _parse_json(self, content: str) -> dict[str, Any]:
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        parsed = json.loads(raw)
        if "route" not in parsed:
            parsed["route"] = "general"
        if "confidence" not in parsed:
            parsed["confidence"] = 0.5
        if "reason" not in parsed:
            parsed["reason"] = "Router model returned a decision."
        return parsed