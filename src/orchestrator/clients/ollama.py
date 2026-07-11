from __future__ import annotations

from typing import Any

import httpx

from ..schemas import ChatMessage, ModelGenerationResponse
from ..settings import Settings


class OllamaClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> ModelGenerationResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "stream": stream,
            "options": options or {},
        }

        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        timeout = httpx.Timeout(self.settings.request_timeout_s)
        close_client = False
        client = self.client
        if client is None:
            client = httpx.AsyncClient(base_url=self.settings.ollama_base_url, timeout=timeout)
            close_client = True

        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            content = data.get("message", {}).get("content", "")
            return ModelGenerationResponse(model=model, content=content, raw=data)
        finally:
            if close_client:
                await client.aclose()