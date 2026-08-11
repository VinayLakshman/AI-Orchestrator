from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ..logging import get_logger
from ..models.chat import ChatMessage
from ..models.ollama import ModelGenerationResponse
from ..settings import Settings

logger = get_logger(__name__)


@dataclass(slots=True)
class LlamaCppStreamChunk:
    content: str
    done: bool
    raw: dict[str, Any]


class LlamaCppClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.base_url = base_url or getattr(settings, "model_router_url", None)
        self.api_key = getattr(settings, "llama_cpp_api_key", None)

    def _build_timeout(self, *, streaming: bool = False) -> httpx.Timeout:
        if streaming:
            return httpx.Timeout(
                connect=self.settings.request_timeout_s,
                read=None,
                write=self.settings.request_timeout_s,
                pool=self.settings.request_timeout_s,
            )
        return httpx.Timeout(self.settings.request_timeout_s)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

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
        keep_alive: str | None = None,  # kept for drop-in compatibility; ignored by llama.cpp
        think: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "stream": stream,
        }

        # Pass through any llama.cpp/OpenAI-compatible extras you may already be using.
        # This keeps the client flexible without hardcoding Ollama-specific payload shape.
        for key, value in (options or {}).items():
            if value is not None:
                payload[key] = value

        if response_format is not None:
            # llama.cpp accepts an explicit response_format object. A bare
            # "json" string is normalized to the OpenAI-compatible shape so
            # callers can keep passing the Ollama-style value unchanged.
            if isinstance(response_format, str) and response_format.lower() == "json":
                payload["response_format"] = {"type": "json_object"}
            else:
                payload["response_format"] = response_format

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # llama.cpp supports reasoning controls on the chat completions endpoint.
        # "none" disables reasoning for this request; anything else is left to the server default.
        if think is not None:
            payload["reasoning_effort"] = "auto" if think else "none"

        # Not used by llama.cpp chat completions; preserved only so callers do not break.
        _ = keep_alive

        return payload

    def _extract_message_content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0]
            if isinstance(choice0, dict):
                message = choice0.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content

        # Fallbacks for atypical server responses.
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        content = data.get("response")
        if isinstance(content, str):
            return content

        return ""

    def _extract_delta_content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0]
            if isinstance(choice0, dict):
                delta = choice0.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str):
                        return content
                    # Some llama.cpp/OpenAI-compatible responses may surface reasoning separately.
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str):
                        return ""
                message = choice0.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
        return ""

    def _log_request(self, *, model: str, think: bool | None) -> None:
        logger.debug(
            "llama_cpp_request model=%s think=%s",
            model,
            "default" if think is None else ("enabled" if think else "disabled"),
        )

    def _log_payload(self, payload: dict[str, Any]) -> None:
        logger.debug("llama_cpp_request_payload=%s", json.dumps(payload, sort_keys=True, default=str))

    async def _get_client(self, *, streaming: bool = False) -> tuple[httpx.AsyncClient, bool]:
        timeout = self._build_timeout(streaming=streaming)
        client = self.client
        close_client = False

        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                headers=self._build_headers(),
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
            content_parts: list[str] = []
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
                    content_parts.append(chunk.content)
                if chunk.raw:
                    final_raw = chunk.raw

            return ModelGenerationResponse(
                model=model,
                content="".join(content_parts).strip(),
                raw=final_raw,
            )

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
        self._log_payload(payload)

        client, close_client = await self._get_client()
        try:
            resp = await client.post("chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = self._extract_message_content(data).strip()
            return ModelGenerationResponse(model=model, content=content, raw=data if isinstance(data, dict) else {})
        except httpx.HTTPStatusError as exc:
            response_text = ""
            with contextlib.suppress(Exception):
                response_text = exc.response.text
            logger.error(
                "llama_cpp_request_failed model=%s status=%s response=%s payload=%s",
                model,
                getattr(exc.response, "status_code", "unknown"),
                response_text[:2000],
                json.dumps(payload, sort_keys=True, default=str),
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
    ) -> AsyncIterator[LlamaCppStreamChunk]:
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
        self._log_payload(payload)

        client, close_client = await self._get_client(streaming=True)
        try:
            async with client.stream("POST", "chat/completions", json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    # OpenAI-compatible streaming uses SSE "data:" lines.
                    if line.startswith("data:"):
                        data_str = line[len("data:") :].strip()
                    else:
                        # Be tolerant of servers that emit plain JSON lines.
                        data_str = line

                    if not data_str:
                        continue

                    if data_str == "[DONE]":
                        yield LlamaCppStreamChunk(content="", done=True, raw={})
                        break

                    try:
                        data = json.loads(data_str)
                    except Exception:
                        continue

                    yield LlamaCppStreamChunk(
                        content=self._extract_delta_content(data),
                        done=bool(data.get("done", False)),
                        raw=data if isinstance(data, dict) else {},
                    )
        except httpx.HTTPStatusError as exc:
            response_text = ""
            with contextlib.suppress(Exception):
                response_text = exc.response.text
            logger.error(
                "llama_cpp_stream_request_failed model=%s status=%s response=%s payload=%s",
                model,
                getattr(exc.response, "status_code", "unknown"),
                response_text[:2000],
                json.dumps(payload, sort_keys=True, default=str),
            )
            raise
        finally:
            if close_client:
                await client.aclose()
