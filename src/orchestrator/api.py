from __future__ import annotations

import asyncio
from contextlib import suppress
from time import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .common.constants import THREAD_ID_MAX_LENGTH
from .graph.build import OrchestratorRuntime
from .models.chat import ChatMessage, ChatRequest
from .models.knowledge import KnowledgeRetrieveResponse
from .models.ollama import ModelGenerationResponse
from .schemas import (
    ControllerPlan,
    ControllerValidation,
    CoderResult,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIMessage,
    OpenAIModelCard,
    OpenAIModelListResponse,
    OrchestratorResponse,
    RouteDecision,
    ToolResult,
)
from .streaming.context import stream_scope
from .streaming.publisher import StreamPublisher
from .streaming.sse import openai_chunk, openai_done

router = APIRouter(tags=["orchestrator"])


def get_runtime(request: Request) -> OrchestratorRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Orchestrator runtime is not ready")
    return runtime


def _request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key in ("authorization", "cookie"):
        value = request.headers.get(key)
        if value:
            headers[key] = value
    return headers


def _thread_id_from_request(payload: ChatRequest) -> str:
    if payload.thread_id:
        return payload.thread_id[:THREAD_ID_MAX_LENGTH]
    return str(uuid4())


def _to_chat_request(
    payload: OpenAIChatCompletionRequest,
    thread_id: str | None = None,
) -> ChatRequest:
    messages = [
        ChatMessage(
            role=msg.role,  # ChatRole-compatible string works here
            content=msg.content,
            name=msg.name,
            tool_call_id=msg.tool_call_id,
        )
        for msg in payload.messages
    ]

    return ChatRequest(
        messages=messages,
        thread_id=thread_id,
        model=payload.model,
        stream=payload.stream,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        metadata=payload.metadata,
    )


def _input_state_from_request(
    payload: ChatRequest,
    request: Request,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    metadata = {
        **payload.metadata,
        "requested_model": payload.model,
        "requested_stream": payload.stream,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "request_headers": _request_headers(request),
    }
    if request_id:
        metadata["request_id"] = request_id

    return {
        "thread_id": _thread_id_from_request(payload),
        "messages": [message.model_dump(exclude_none=True) for message in payload.messages],
        "metadata": metadata,
        "used_models": [],
        "used_tools": [],
        "knowledge_result": None,
        "vision": None,
        "vision_context": "",
        "coder_result": None,
        "tool_result": None,
        "reasoning_result": None,
        "pending_steps": [],
        "completed_steps": [],
        "current_step": "",
        "needs_reasoning": False,
        "requires_clarification": False,
        "controller_cycles": 0,
        "specialist_executions": 0,
        "workflow_stall_count": 0,
        "last_progress_signature": "",
        "final_answer_ready": False,
    }


def _route_from_state(state: dict[str, Any]) -> RouteDecision | None:
    route = state.get("route")
    if route is None:
        plan = state.get("controller_plan")
        if isinstance(plan, dict):
            route = plan.get("route_hint")
    if route is None:
        return None
    return RouteDecision.model_validate(route)


def _knowledge_from_state(state: dict[str, Any]) -> KnowledgeRetrieveResponse | None:
    knowledge_result = state.get("knowledge_result")
    if isinstance(knowledge_result, dict):
        return KnowledgeRetrieveResponse.model_validate(knowledge_result)
    if isinstance(knowledge_result, KnowledgeRetrieveResponse):
        return knowledge_result
    return None


def _controller_plan_from_state(state: dict[str, Any]) -> ControllerPlan | None:
    value = state.get("controller_plan")
    if isinstance(value, dict):
        return ControllerPlan.model_validate(value)
    if isinstance(value, ControllerPlan):
        return value
    return None


def _controller_validation_from_state(state: dict[str, Any]) -> ControllerValidation | None:
    value = state.get("controller_validation")
    if isinstance(value, dict):
        return ControllerValidation.model_validate(value)
    if isinstance(value, ControllerValidation):
        return value
    return None


def _coder_from_state(state: dict[str, Any]) -> CoderResult | None:
    value = state.get("coder_result")
    if isinstance(value, dict):
        return CoderResult.model_validate(value)
    if isinstance(value, CoderResult):
        return value
    return None


def _tool_from_state(state: dict[str, Any]) -> ToolResult | None:
    value = state.get("tool_result")
    if isinstance(value, dict):
        return ToolResult.model_validate(value)
    if isinstance(value, ToolResult):
        return value
    return None


def _assistant_content_from_messages(state: dict[str, Any]) -> str:
    messages = state.get("messages", []) or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if content is not None:
                text = str(content).strip()
                if text:
                    return text
    return ""


def _answer_from_state(state: dict[str, Any]) -> tuple[str, str]:
    answer = str(state.get("answer", "") or "").strip()
    if answer:
        return answer, "answer"

    reasoning = state.get("reasoning_result")
    if isinstance(reasoning, dict):
        reasoning = ModelGenerationResponse.model_validate(reasoning)
    if isinstance(reasoning, ModelGenerationResponse) and reasoning.content.strip():
        return reasoning.content.strip(), "reasoning_result.content"

    assistant_content = _assistant_content_from_messages(state)
    if assistant_content:
        return assistant_content, "messages[-1].content"

    metadata = state.get("metadata", {}) or {}
    if isinstance(metadata, dict):
        final_answer = str(metadata.get("final_answer", "") or "").strip()
        if final_answer:
            return final_answer, "metadata.final_answer"

    return "", "missing"


def _response_from_state(thread_id: str, state: dict[str, Any]) -> OrchestratorResponse:
    answer, _answer_source = _answer_from_state(state)
    reasoning = state.get("reasoning_result")
    if isinstance(reasoning, dict):
        reasoning = ModelGenerationResponse.model_validate(reasoning)
    elif reasoning is not None and not isinstance(reasoning, ModelGenerationResponse):
        reasoning = None

    return OrchestratorResponse(
        thread_id=thread_id,
        route=_route_from_state(state),
        controller_plan=_controller_plan_from_state(state),
        controller_validation=_controller_validation_from_state(state),
        answer=answer,
        used_models=state.get("used_models", []),
        used_tools=state.get("used_tools", []),
        knowledge_result=_knowledge_from_state(state),
        vision=state.get("vision"),
        vision_context=state.get("vision_context", ""),
        coder_result=_coder_from_state(state),
        tool_result=_tool_from_state(state),
        reasoning=reasoning,
        metadata=state.get("metadata", {}),
    )


async def _run_graph_with_stream(
    *,
    runtime: OrchestratorRuntime,
    request_id: str,
    thread_id: str,
    state_input: dict[str, Any],
    publisher: StreamPublisher,
) -> dict[str, Any]:
    async with stream_scope(publisher):
        await publisher.graph_started()
        try:
            result = await runtime.graph.ainvoke(
                state_input,
                config={"configurable": {"thread_id": thread_id}},
            )
            await publisher.graph_finished(route=(result.get("route") or {}).get("route") if isinstance(result.get("route"), dict) else None)
            return result
        except Exception as exc:
            await publisher.graph_failed(str(exc))
            raise


def _request_headers_out(request_id: str, thread_id: str) -> dict[str, str]:
    return {
        "cache-control": "no-cache",
        "connection": "keep-alive",
        "x-accel-buffering": "no",
        "x-orchestrator-request-id": request_id,
        "x-orchestrator-thread-id": thread_id,
    }


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(runtime: OrchestratorRuntime = Depends(get_runtime)) -> dict[str, str]:
    return {
        "status": "ready",
        "graph": "compiled",
        "checkpointer": runtime.checkpointer.__class__.__name__,
    }


@router.get("/v1/models", response_model=OpenAIModelListResponse)
async def list_models(runtime: OrchestratorRuntime = Depends(get_runtime)) -> OpenAIModelListResponse:
    settings = runtime.settings
    return OpenAIModelListResponse(
        data=[
            OpenAIModelCard(id="orchestrator", owned_by="local"),
            OpenAIModelCard(id=settings.controller_model, owned_by="local"),
            OpenAIModelCard(id=settings.reasoning_model, owned_by="local"),
            OpenAIModelCard(id=settings.coder_model, owned_by="local"),
            OpenAIModelCard(id=settings.vision_model, owned_by="local"),
            OpenAIModelCard(id=settings.embedding_model, owned_by="local"),
        ]
    )


@router.post("/chat", response_model=OrchestratorResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    runtime: OrchestratorRuntime = Depends(get_runtime),
) -> OrchestratorResponse:
    thread_id = _thread_id_from_request(payload)
    state_input = _input_state_from_request(payload, request)

    result = await runtime.graph.ainvoke(
        state_input,
        config={"configurable": {"thread_id": thread_id}},
    )

    return _response_from_state(thread_id, result)


@router.get("/v1/streams/{request_id}")
async def stream_events(
    request: Request,
    request_id: str,
    runtime: OrchestratorRuntime = Depends(get_runtime),
) -> StreamingResponse:
    stream = await runtime.stream_hub.get(request_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Unknown request_id")

    after_seq_raw = request.headers.get("last-event-id") or request.query_params.get("after_seq") or "0"
    try:
        after_seq = int(after_seq_raw)
    except ValueError:
        after_seq = 0

    async def event_gen():
        async for event in stream.subscribe(after_seq=after_seq):
            yield event.to_sse()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers=_request_headers_out(request_id, stream.conversation_id or ""),
    )


@router.post(
    "/v1/chat/completions",
    response_model=OpenAIChatCompletionResponse,
)
async def openai_chat_completions(
    payload: OpenAIChatCompletionRequest,
    request: Request,
    runtime: OrchestratorRuntime = Depends(get_runtime),
):
    request_id = str(uuid4())
    thread_id = _thread_id_from_request(_to_chat_request(payload))
    chat_request = _to_chat_request(payload, thread_id=thread_id)
    state_input = _input_state_from_request(chat_request, request, request_id=request_id)

    stream = runtime.stream_hub.get_or_create(request_id, conversation_id=thread_id)
    publisher = StreamPublisher(stream)

    if payload.stream:
        async def sse_generator():
            result: dict[str, Any] | None = None
            try:
                result = await _run_graph_with_stream(
                    runtime=runtime,
                    request_id=request_id,
                    thread_id=thread_id,
                    state_input=state_input,
                    publisher=publisher,
                )
                answer = str((result or {}).get("answer", "") or "")
                if answer:
                    yield openai_chunk(
                        id=request_id,
                        model=str(payload.model),
                        content=answer,
                        request_id=request_id,
                    )
                yield openai_done()
            finally:
                await stream.close()

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers=_request_headers_out(request_id, thread_id),
        )

    result = await runtime.graph.ainvoke(
        state_input,
        config={"configurable": {"thread_id": thread_id}},
    )

    answer, _answer_source = _answer_from_state(result or {})
    completion = OpenAIChatCompletionResponse(
        id=request_id,
        created=int(time()),
        model=str(payload.model),
        choices=[
            OpenAIChatCompletionChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
        metadata={
            "thread_id": thread_id,
            "request_id": request_id,
        },
    )

    await stream.close()

    return completion
