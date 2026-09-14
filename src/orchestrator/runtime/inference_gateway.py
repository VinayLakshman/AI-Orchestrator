from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..clients.llama_cpp import LlamaCppClient, LlamaCppStreamChunk
from ..models.generation import ModelGenerationResponse


@dataclass(slots=True)
class ManagedInferenceClient:
    """Role-scoped inference client with centralized GPU scheduling.

    The wrapped transport is shared by all model roles. The lifecycle manager
    owns the single active-model lease, so every call path—including streaming
    calls—passes through the same arbitration boundary.
    """

    role: str
    transport: LlamaCppClient
    lifecycle: Any
    model_name: str

    async def chat(self, model: str, messages: list[Any], **kwargs: Any) -> ModelGenerationResponse:
        del model
        async with self.lifecycle.llm_inference(self.role):
            return await self.transport.chat(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )

    async def stream_chat(
        self,
        model: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> AsyncIterator[LlamaCppStreamChunk]:
        del model
        async with self.lifecycle.llm_inference(self.role):
            async for chunk in self.transport.stream_chat(
                model=self.model_name,
                messages=messages,
                **kwargs,
            ):
                yield chunk

    async def aclose(self) -> None:
        """Managed role views do not own the shared transport."""

    @property
    def underlying_client(self) -> LlamaCppClient:
        return self.transport


__all__ = ["ManagedInferenceClient"]
