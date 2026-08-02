from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any
import json

import httpx

from ..models.ollama import (
    ModelGenerationResponse,
    OllamaStreamChunk,
    extract_assistant_text,
    normalize_generation_response,
)
from ..logging import get_logger
from ..settings import Settings
from ..models.chat import ChatMessage
from ..serialization import sanitize_for_json, validate_json_serializable, SerializationError


logger = get_logger(__name__)


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
        response_format: str | dict[str, Any] | None = None,
        keep_alive: str | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "stream": stream,
            "options": dict(options or {}),
        }

        if response_format is not None:
            payload["format"] = response_format

        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if think is not None:
            payload["think"] = think

        return payload

    def _extract_content(self, data: dict[str, Any]) -> str:
        """
        Extract a single streamed token from an Ollama streaming response.

        This intentionally performs no normalization or whitespace stripping.
        Streaming must preserve the model output exactly.
        """
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        content = data.get("response")
        if isinstance(content, str):
            return content

        return ""

    def _log_request(self, *, model: str, think: bool | None) -> None:
        logger.debug(
            "ollama_request model=%s think=%s",
            model,
            "default" if think is None else ("enabled" if think else "disabled"),
        )

    def _log_payload(self, payload: dict[str, Any]) -> None:
        try:
            logger.debug("ollama_request_payload=%s", json.dumps(payload, sort_keys=True, ensure_ascii=False))
        except Exception:
            # Best-effort fallback to repr if something unexpected slipped through
            logger.debug("ollama_request_payload_repr=%s", repr(payload))

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
        response_format: str | dict[str, Any] | None = None,
        keep_alive: str | None = None,
        think: bool | None = None,
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
                response_format=response_format,
                keep_alive=keep_alive,
                think=think,
            ):
                if chunk.content:
                    content.append(chunk.content)
                final_raw = chunk.raw or final_raw
            return ModelGenerationResponse(model=model, content="".join(content).strip(), raw=final_raw)

        self._log_request(model=model, think=think)
        payload = self._build_payload(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            options=options,
            response_format=response_format,
            keep_alive=keep_alive,
            think=think,
        )
        # Sanitize and validate payload to ensure JSON-compatibility
        try:
            sanitized_payload = sanitize_for_json(payload)
            validate_json_serializable(sanitized_payload)
        except SerializationError as exc:
            logger.exception("ollama_payload_serialization_error model=%s error=%s payload_preview=%s", model, exc, str(payload)[:1000])
            raise

        self._log_payload(sanitized_payload)
        client, close_client = await self._get_client()
        try:
            resp = await client.post("/api/chat", json=sanitized_payload)
            resp.raise_for_status()
            data = resp.json()
            return normalize_generation_response(model, data)
        except httpx.HTTPStatusError as exc:
            response_text = ""
            with contextlib.suppress(Exception):
                response_text = exc.response.text
            logger.error(
                "ollama_request_failed model=%s status=%s response=%s payload=%s",
                model,
                getattr(exc.response, "status_code", "unknown"),
                response_text[:2000],
                json.dumps(sanitized_payload, sort_keys=True, ensure_ascii=False),
            )
            raise
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
        response_format: str | dict[str, Any] | None = None,
        keep_alive: str | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[OllamaStreamChunk]:
        self._log_request(model=model, think=think)
        payload = self._build_payload(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            options=options,
            response_format=response_format,
            keep_alive=keep_alive,
            think=think,
        )
        # Sanitize and validate payload for streaming
        try:
            sanitized_payload = sanitize_for_json(payload)
            validate_json_serializable(sanitized_payload)
        except SerializationError as exc:
            logger.exception("ollama_stream_payload_serialization_error model=%s error=%s payload_preview=%s", model, exc, str(payload)[:1000])
            raise

        self._log_payload(sanitized_payload)

        client, close_client = await self._get_client(streaming=True)
        try:
            async with client.stream("POST", "/api/chat", json=sanitized_payload) as resp:
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
        except httpx.HTTPStatusError as exc:
            response_text = ""
            with contextlib.suppress(Exception):
                response_text = exc.response.text
            logger.error(
                "ollama_stream_request_failed model=%s status=%s response=%s payload=%s",
                model,
                getattr(exc.response, "status_code", "unknown"),
                response_text[:2000],
                json.dumps(sanitized_payload, sort_keys=True, ensure_ascii=False),
            )
            raise
        finally:
            if close_client:
                await client.aclose()
