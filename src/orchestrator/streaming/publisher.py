from __future__ import annotations

from typing import Any

from .hub import RequestEventStream
from .models import StreamEvent, StreamKind


class StreamPublisher:
    def __init__(self, request_stream: RequestEventStream) -> None:
        self.stream = request_stream

    async def emit(
        self,
        *,
        kind: StreamKind,
        message: str,
        stage: str | None = None,
        status: str = "info",
        data: dict[str, Any] | None = None,
    ) -> StreamEvent:
        event = StreamEvent(
            request_id=self.stream.request_id,
            conversation_id=self.stream.conversation_id,
            kind=kind,
            stage=stage,
            message=message,
            status=status,  # type: ignore[arg-type]
            data=data or {},
        )
        return await self.stream.publish(event)

    async def graph_started(self, *, route_hint: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {}
        if route_hint:
            payload["route_hint"] = route_hint
        return await self.emit(
            kind=StreamKind.GRAPH_STARTED,
            stage="orchestration",
            message="Starting request orchestration.",
            status="progress",
            data=payload,
        )

    async def graph_finished(self, *, route: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {}
        if route:
            payload["route"] = route
        return await self.emit(
            kind=StreamKind.GRAPH_FINISHED,
            stage="orchestration",
            message="Request completed.",
            status="success",
            data=payload,
        )

    async def graph_failed(self, error: str) -> StreamEvent:
        return await self.emit(
            kind=StreamKind.GRAPH_FAILED,
            stage="orchestration",
            message="Request failed.",
            status="error",
            data={"error": error},
        )

    async def routing_started(self, *, query: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {}
        if query:
            payload["query"] = query
        return await self.emit(
            kind=StreamKind.ROUTING_STARTED,
            stage="routing",
            message="Analyzing the request and selecting a route.",
            status="progress",
            data=payload,
        )

    async def routing_finished(self, *, route: str, reason: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {"route": route}
        if reason:
            payload["reason"] = reason
        return await self.emit(
            kind=StreamKind.ROUTING_FINISHED,
            stage="routing",
            message=f"Route selected: {route}.",
            status="success",
            data=payload,
        )

    async def knowledge_started(self, *, query: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {}
        if query:
            payload["query"] = query
        return await self.emit(
            kind=StreamKind.KNOWLEDGE_STARTED,
            stage="retrieval",
            message="Searching the knowledge base.",
            status="progress",
            data=payload,
        )

    async def knowledge_progress(self, *, message: str, data: dict[str, Any] | None = None) -> StreamEvent:
        return await self.emit(
            kind=StreamKind.KNOWLEDGE_PROGRESS,
            stage="retrieval",
            message=message,
            status="progress",
            data=data or {},
        )

    async def knowledge_finished(
        self,
        *,
        documents: int,
        sources: list[str] | None = None,
    ) -> StreamEvent:
        payload: dict[str, Any] = {"documents": documents}
        if sources:
            payload["sources"] = sources
        return await self.emit(
            kind=StreamKind.KNOWLEDGE_FINISHED,
            stage="retrieval",
            message=f"Retrieved {documents} knowledge documents.",
            status="success",
            data=payload,
        )

    async def vision_started(self, *, image_count: int = 1) -> StreamEvent:
        return await self.emit(
            kind=StreamKind.VISION_STARTED,
            stage="vision",
            message="Analyzing image input.",
            status="progress",
            data={"image_count": image_count},
        )

    async def vision_progress(self, *, message: str, data: dict[str, Any] | None = None) -> StreamEvent:
        return await self.emit(
            kind=StreamKind.VISION_PROGRESS,
            stage="vision",
            message=message,
            status="progress",
            data=data or {},
        )

    async def vision_finished(self, *, summary: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {}
        if summary:
            payload["summary"] = summary
        return await self.emit(
            kind=StreamKind.VISION_FINISHED,
            stage="vision",
            message="Image analysis completed.",
            status="success",
            data=payload,
        )

    async def tool_started(self, *, tool_name: str, tool_type: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {"tool_name": tool_name}
        if tool_type:
            payload["tool_type"] = tool_type
        return await self.emit(
            kind=StreamKind.TOOL_STARTED,
            stage="tools",
            message=f"Calling tool: {tool_name}.",
            status="progress",
            data=payload,
        )

    async def tool_progress(self, *, tool_name: str, message: str, data: dict[str, Any] | None = None) -> StreamEvent:
        payload: dict[str, Any] = {"tool_name": tool_name}
        if data:
            payload.update(data)
        return await self.emit(
            kind=StreamKind.TOOL_PROGRESS,
            stage="tools",
            message=message,
            status="progress",
            data=payload,
        )

    async def tool_finished(self, *, tool_name: str, result_preview: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {"tool_name": tool_name}
        if result_preview:
            payload["result_preview"] = result_preview
        return await self.emit(
            kind=StreamKind.TOOL_FINISHED,
            stage="tools",
            message=f"Tool completed: {tool_name}.",
            status="success",
            data=payload,
        )

    async def code_started(self, *, task: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {}
        if task:
            payload["task"] = task
        return await self.emit(
            kind=StreamKind.CODE_STARTED,
            stage="coding",
            message="Working on code generation.",
            status="progress",
            data=payload,
        )

    async def code_progress(self, *, message: str, data: dict[str, Any] | None = None) -> StreamEvent:
        return await self.emit(
            kind=StreamKind.CODE_PROGRESS,
            stage="coding",
            message=message,
            status="progress",
            data=data or {},
        )

    async def code_finished(self, *, result: str | None = None) -> StreamEvent:
        payload: dict[str, Any] = {}
        if result:
            payload["result_preview"] = result
        return await self.emit(
            kind=StreamKind.CODE_FINISHED,
            stage="coding",
            message="Coding step completed.",
            status="success",
            data=payload,
        )

    async def llm_started(self, *, model: str) -> StreamEvent:
        return await self.emit(
            kind=StreamKind.LLM_STARTED,
            stage="generation",
            message="Generating final response.",
            status="progress",
            data={"model": model},
        )

    async def llm_token(self, token: str) -> StreamEvent:
        return await self.emit(
            kind=StreamKind.LLM_TOKEN,
            stage="generation",
            message="Generated token.",
            status="progress",
            data={"token": token},
        )

    async def llm_finished(self) -> StreamEvent:
        return await self.emit(
            kind=StreamKind.LLM_FINISHED,
            stage="generation",
            message="Response generation finished.",
            status="success",
        )

    async def error(self, error: str, *, stage: str | None = None) -> StreamEvent:
        return await self.emit(
            kind=StreamKind.ERROR,
            stage=stage,
            message=error,
            status="error",
            data={"error": error},
        )