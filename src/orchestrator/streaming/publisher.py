from __future__ import annotations

from typing import Any

from .hub import RequestEventStream
from .models import StreamEvent, StreamKind


def _clean_payload(**payload: Any) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


class StreamPublisher:
    def __init__(self, request_stream: RequestEventStream) -> None:
        self.stream = request_stream

    async def _emit(self, kind: StreamKind, **payload: Any) -> StreamEvent:
        return await self.stream.publish(kind, **payload)

    async def graph_started(
        self,
        *,
        route_hint: str | None = None,
        message: str = "Starting request orchestration.",
    ) -> StreamEvent:
        return await self._emit(
            StreamKind.GRAPH_STARTED,
            **_clean_payload(
                stage="orchestration",
                message=message,
                route_hint=route_hint,
            ),
        )

    async def graph_finished(
        self,
        *,
        route: str | None = None,
        message: str = "Request completed.",
    ) -> StreamEvent:
        return await self._emit(
            StreamKind.GRAPH_FINISHED,
            **_clean_payload(
                stage="orchestration",
                message=message,
                route=route,
            ),
        )

    async def graph_failed(self, error: str) -> StreamEvent:
        return await self._emit(
            StreamKind.GRAPH_FAILED,
            stage="orchestration",
            message=error,
            error=error,
        )

    async def specialist_started(
        self,
        *,
        specialist: str,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> StreamEvent:
        return await self._emit(
            StreamKind.SPECIALIST_STARTED,
            **_clean_payload(
                specialist=specialist,
                message=message or f"{specialist} started.",
                data=data or {},
            ),
        )

    async def specialist_progress(
        self,
        *,
        specialist: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> StreamEvent:
        return await self._emit(
            StreamKind.SPECIALIST_PROGRESS,
            **_clean_payload(
                specialist=specialist,
                message=message,
                data=data or {},
            ),
        )

    async def specialist_finished(
        self,
        *,
        specialist: str,
        success: bool = True,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> StreamEvent:
        return await self._emit(
            StreamKind.SPECIALIST_FINISHED,
            **_clean_payload(
                specialist=specialist,
                status="success" if success else "failed",
                message=message or (
                    f"{specialist} completed." if success else f"{specialist} failed."
                ),
                data=data or {},
            ),
        )

    async def llm_started(
        self,
        *,
        model: str,
        stage: str = "generation",
        message: str = "Generating response.",
        data: dict[str, Any] | None = None,
    ) -> StreamEvent:
        payload = {"model": model, "stage": stage, "message": message}
        if data:
            payload["data"] = data
        return await self._emit(StreamKind.LLM_STARTED, **payload)

    async def llm_token(self, token: str) -> StreamEvent:
        return await self._emit(
            StreamKind.LLM_TOKEN,
            stage="generation",
            message="Generated token.",
            token=token,
        )

    async def llm_finished(
        self,
        *,
        stage: str = "generation",
        message: str = "Generation completed.",
        data: dict[str, Any] | None = None,
    ) -> StreamEvent:
        payload = {"stage": stage, "message": message}
        if data:
            payload["data"] = data
        return await self._emit(StreamKind.LLM_FINISHED, **payload)

    async def routing_started(self, *, query: str | None = None) -> StreamEvent:
        return await self.specialist_started(
            specialist="routing",
            message="Analyzing the request and selecting a route.",
            data={"query": query} if query else {},
        )

    async def routing_finished(
        self,
        *,
        route: str,
        reason: str | None = None,
    ) -> StreamEvent:
        data = {"route": route}
        if reason:
            data["reason"] = reason
        return await self.specialist_finished(
            specialist="routing",
            success=True,
            message="Routing complete.",
            data=data,
        )

    async def controller_started(self, *, step: str) -> StreamEvent:
        return await self.specialist_started(
            specialist="controller",
            message=f"Controller {step} started.",
            data={"step": step},
        )

    async def controller_plan(self, *, intent: str, steps: list[str]) -> StreamEvent:
        return await self.specialist_progress(
            specialist="controller",
            message="Controller produced a plan.",
            data={"intent": intent, "steps": steps},
        )

    async def controller_validated(
        self,
        *,
        action: str,
        issues: list[str],
    ) -> StreamEvent:
        return await self.specialist_finished(
            specialist="controller",
            success=True,
            message="Controller validation complete.",
            data={"action": action, "issues": issues},
        )

    async def validation_started(
        self,
        *,
        message: str | None = None,
    ) -> StreamEvent:
        return await self.specialist_started(
            specialist="validation",
            message=message or "Validating the result...",
            data={},
        )

    async def knowledge_started(self, *, query: str | None = None) -> StreamEvent:
        return await self.specialist_started(
            specialist="knowledge",
            message="Searching the knowledge base.",
            data={"query": query} if query else {},
        )

    async def knowledge_finished(
        self,
        *,
        documents: int,
        sources: list[str] | None = None,
    ) -> StreamEvent:
        data: dict[str, Any] = {"documents": documents}
        if sources:
            data["sources"] = sources
        return await self.specialist_finished(
            specialist="knowledge",
            success=documents > 0,
            message="Knowledge retrieval completed.",
            data=data,
        )

    async def web_search_started(self, *, query: str | None = None) -> StreamEvent:
        return await self.specialist_started(
            specialist="web",
            message="Searching the web.",
            data={"query": query} if query else {},
        )

    async def web_search_processing(self, *, results: int) -> StreamEvent:
        return await self.specialist_progress(
            specialist="web",
            message="Processing web search results.",
            data={"results": results},
        )

    async def web_search_finished(
        self,
        *,
        results: int,
        search_time_ms: int,
    ) -> StreamEvent:
        return await self.specialist_finished(
            specialist="web",
            success=results > 0,
            message="Web search completed.",
            data={"results": results, "search_time_ms": search_time_ms},
        )

    async def vision_started(self, *, image_count: int) -> StreamEvent:
        return await self.specialist_started(
            specialist="vision",
            message="Starting vision analysis.",
            data={"image_count": image_count},
        )

    async def vision_progress(
        self,
        *,
        message: str,
        data: dict[str, Any],
    ) -> StreamEvent:
        return await self.specialist_progress(
            specialist="vision",
            message=message,
            data=data,
        )

    async def vision_finished(self, *, summary: str | None = None) -> StreamEvent:
        data: dict[str, Any] = {}
        if summary:
            data["summary"] = summary
        return await self.specialist_finished(
            specialist="vision",
            success=True,
            message="Image analysis completed.",
            data=data,
        )

    async def tool_started(
        self,
        *,
        tool_name: str,
        tool_type: str | None = None,
    ) -> StreamEvent:
        data: dict[str, Any] = {"tool_name": tool_name}
        if tool_type:
            data["tool_type"] = tool_type
        return await self.specialist_started(
            specialist="tools",
            message=f"Starting tool execution for {tool_name}.",
            data=data,
        )

    async def tool_progress(
        self,
        *,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> StreamEvent:
        return await self.specialist_progress(
            specialist="tools",
            message=message,
            data=data,
        )

    async def tool_finished(
        self,
        *,
        tool_name: str,
        success: bool = True,
        result: dict[str, Any] | None = None,
    ) -> StreamEvent:
        data: dict[str, Any] = {"tool_name": tool_name}
        if result is not None:
            data["result"] = result
        return await self.specialist_finished(
            specialist="tools",
            success=success,
            message="Tool execution completed.",
            data=data,
        )

    async def code_started(
        self,
        *,
        model: str | None = None,
        task: str | None = None,
    ) -> StreamEvent:
        data: dict[str, Any] = {}
        if model:
            data["model"] = model
        if task:
            data["task"] = task
        return await self.specialist_started(
            specialist="code",
            message="Working on code generation.",
            data=data,
        )

    async def code_progress(
        self,
        *,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> StreamEvent:
        return await self.specialist_progress(
            specialist="code",
            message=message,
            data=data,
        )

    async def code_finished(self, *, result: str | None = None) -> StreamEvent:
        data: dict[str, Any] = {}
        if result:
            data["result_preview"] = result
        return await self.specialist_finished(
            specialist="code",
            success=True,
            message="Coding step completed.",
            data=data,
        )

    async def reasoning_started(self, *, model: str) -> StreamEvent:
        return await self.llm_started(
            model=model,
            stage="reasoning",
            message="Reasoning started.",
        )

    async def reasoning_token(self, token: str) -> StreamEvent:
        return await self.llm_token(token)

    async def reasoning_finished(self) -> StreamEvent:
        return await self.llm_finished(
            stage="reasoning",
            message="Reasoning completed.",
        )

    async def image_generation_started(
        self,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> StreamEvent:
        return await self._emit(
            StreamKind.IMAGE_GENERATION_STARTED,
            **_clean_payload(
                specialist="image_generation",
                message=message or "Image generation started.",
                data=data or {},
            ),
        )

    async def image_generation_finished(
        self,
        *,
        success: bool = True,
        image_url: str | None = None,
        error: str | None = None,
        message: str | None = None,
    ) -> StreamEvent:
        data: dict[str, Any] = {}
        if image_url is not None:
            data["image_url"] = image_url
        if error is not None:
            data["error"] = error
        
        return await self._emit(
            StreamKind.IMAGE_GENERATION_FINISHED,
            **_clean_payload(
                specialist="image_generation",
                status="success" if success else "failed",
                message=message or (
                    "Image generation completed." if success else "Image generation failed."
                ),
                data=data or {},
            ),
        )