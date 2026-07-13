from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
import json

import httpx

from ..schemas import ModelGenerationResponse
from ..settings import Settings
from ..models.chat import ChatMessage


@dataclass(slots=True)
class OllamaStreamChunk:
    content: str
    done: bool
    raw: dict[str, Any]


class OllamaClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    def _build_timeout(self, *, streaming: bool = False) -> httpx.Timeout:
        if streaming:
            return httpx.Timeout(
                connect=self.settings.request_timeout_s,
                read=None,
                write=self.settings.request_timeout_s,
                pool=self.settings.request_timeout_s,
            )
        return httpx.Timeout(self.settings.request_timeout_s)

    def _build_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "stream": stream,
            "options": dict(options or {}),
        }

        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        return payload

    def _extract_content(self, data: dict[str, Any]) -> str:
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                return str(content)

        if "response" in data and data["response"] is not None:
            return str(data.get("response") or "")

        if "content" in data:
            content = data["content"]
            if isinstance(content, list):
                return "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ).strip()
            return str(content or "")

        if "choices" in data and data["choices"]:
            try:
                choice = data["choices"][0]
                return str(choice.get("message", {}).get("content", "") or "")
            except Exception:
                return ""

        return ""

    async def _get_client(self, *, streaming: bool = False) -> tuple[httpx.AsyncClient, bool]:
        timeout = self._build_timeout(streaming=streaming)
        client = self.client
        close_client = False
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=timeout,
            )
            close_client = True
        return client, close_client

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> ModelGenerationResponse:
        if stream:
            content = []
            final_raw: dict[str, Any] = {}
            async for chunk in self.stream_chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                options=options,
                keep_alive=keep_alive,
            ):
                if chunk.content:
                    content.append(chunk.content)
                final_raw = chunk.raw or final_raw
            return ModelGenerationResponse(model=model, content="".join(content).strip(), raw=final_raw)

        payload = self._build_payload(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            options=options,
            keep_alive=keep_alive,
        )

        client, close_client = await self._get_client()
        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = self._extract_content(data)
            return ModelGenerationResponse(model=model, content=content, raw=data if isinstance(data, dict) else {})
        finally:
            if close_client:
                await client.aclose()

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> AsyncIterator[OllamaStreamChunk]:
        payload = self._build_payload(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            options=options,
            keep_alive=keep_alive,
        )

        client, close_client = await self._get_client(streaming=True)
        try:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue

                    yield OllamaStreamChunk(
                        content=self._extract_content(data),
                        done=bool(data.get("done", False)),
                        raw=data if isinstance(data, dict) else {},
                    )
        finally:
            if close_client:
                await client.aclose()