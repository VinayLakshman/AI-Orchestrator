from __future__ import annotations

from typing import Any

from .hub import RequestEventStream
from .models import StreamEvent, StreamKind


class StreamPublisher:
    def __init__(self, stream: RequestEventStream) -> None:
        self.stream = stream

    async def _emit(self, kind: StreamKind, **payload: Any) -> None:
        await self.stream.publish(kind, **payload)

    async def graph_started(self) -> None:
        await self._emit(StreamKind.GRAPH_STARTED)

    async def graph_finished(self, route: str | None = None) -> None:
        await self._emit(StreamKind.GRAPH_FINISHED, route=route)

    async def graph_failed(self, error: str) -> None:
        await self._emit(StreamKind.GRAPH_FAILED, error=error)

    async def controller_started(self, *, step: str) -> None:
        await self._emit(StreamKind.CONTROLLER_STARTED, step=step)

    async def controller_plan(self, *, intent: str, steps: list[str]) -> None:
        await self._emit(StreamKind.CONTROLLER_PLAN, intent=intent, steps=steps)

    async def controller_validated(self, *, action: str, issues: list[str]) -> None:
        await self._emit(StreamKind.CONTROLLER_VALIDATED, action=action, issues=issues)

    async def knowledge_started(self, *, query: str) -> None:
        await self._emit(StreamKind.KNOWLEDGE_STARTED, query=query)

    async def knowledge_finished(self, *, documents: int, sources: list[str]) -> None:
        await self._emit(StreamKind.KNOWLEDGE_FINISHED, documents=documents, sources=sources)

    async def vision_started(self, *, image_count: int) -> None:
        await self._emit(StreamKind.VISION_STARTED, image_count=image_count)

    async def vision_progress(self, *, message: str, data: dict[str, Any]) -> None:
        await self._emit(StreamKind.VISION_PROGRESS, message=message, data=data)

    async def vision_finished(self, *, summary: str) -> None:
        await self._emit(StreamKind.VISION_FINISHED, summary=summary)

    async def code_started(self, *, model: str) -> None:
        await self._emit(StreamKind.CODE_STARTED, model=model)

    async def code_finished(self, *, result: str) -> None:
        await self._emit(StreamKind.CODE_FINISHED, result=result)

    async def reasoning_started(self, *, model: str) -> None:
        await self._emit(StreamKind.REASONING_STARTED, model=model)

    async def reasoning_token(self, token: str) -> None:
        await self._emit(StreamKind.REASONING_TOKEN, token=token)

    async def reasoning_finished(self) -> None:
        await self._emit(StreamKind.REASONING_FINISHED)

    async def llm_started(self, *, model: str) -> None:
        await self._emit(StreamKind.LLM_STARTED, model=model)

    async def llm_finished(self) -> None:
        await self._emit(StreamKind.LLM_FINISHED)

    async def llm_token(self, token: str) -> StreamEvent:
        return await self._emit(
            kind=StreamKind.LLM_TOKEN,
            stage="generation",
            message="Generated token.",
            status="progress",
            data={"token": token},
        )

    async def error(self, error: str, *, stage: str | None = None) -> None:
        await self._emit(StreamKind.ERROR, error=error, stage=stage)