"""Translate HTTP payloads and graph state without owning route behavior."""

from __future__ import annotations

from time import time
from typing import Any
from uuid import uuid4

from fastapi import Request

from ..common.constants import FALLBACK_NO_ANSWER, THREAD_ID_MAX_LENGTH
from ..graph.build import OrchestratorRuntime
from ..models.chat import ChatRequest
from ..models.state import OrchestratorState, RequestState
from ..schemas import (
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIMessage,
    OpenAIUsage,
)


def request_headers(request_id: str, thread_id: str) -> dict[str, str]:
    return {
        "cache-control": "no-cache",
        "connection": "keep-alive",
        "x-accel-buffering": "no",
        "x-orchestrator-request-id": request_id,
        "x-orchestrator-thread-id": thread_id,
    }


def native_thread_id(payload: ChatRequest) -> str:
    if payload.thread_id:
        return payload.thread_id[:THREAD_ID_MAX_LENGTH]
    return str(uuid4())


def openai_thread_id(request: Request, payload: OpenAIChatCompletionRequest) -> str:
    """Resolve an optional stable conversation identity for OpenAI clients."""
    header_value = request.headers.get("x-orchestrator-thread-id")
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    metadata_value = metadata.get("thread_id")
    candidate = str(header_value or metadata_value or "").strip()
    return candidate[:THREAD_ID_MAX_LENGTH] if candidate else str(uuid4())


def openai_request_from_chat_request(payload: ChatRequest) -> OpenAIChatCompletionRequest:
    return OpenAIChatCompletionRequest(
        model=payload.model or "orchestrator",
        messages=[
            OpenAIMessage(
                role=message.role,
                content=message.content,
                name=message.name,
                tool_call_id=message.tool_call_id,
            )
            for message in payload.messages
        ],
        stream=payload.stream,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        params=payload.params,
        metadata=payload.metadata,
    )


def input_state_from_request_state(
    request_state: RequestState,
    *,
    thread_id: str,
    request_id: str | None = None,
    model: str = "orchestrator",
    stream: bool = False,
) -> OrchestratorState:
    request_id = request_id or str(uuid4())
    request_state = request_state.model_copy(
        update={
            "request_id": request_id,
            "conversation_id": thread_id,
            "thread_id": thread_id,
            "model": model,
            "stream": stream,
            "metadata": {
                **request_state.metadata,
                "request_headers": request_headers(request_id, thread_id),
            },
        }
    )
    return OrchestratorState(request=request_state)


def graph_input(state_input: OrchestratorState) -> dict[str, Any]:
    """Pass only request state so checkpointed conversation state survives."""
    return {"request": state_input.request}


def image_urls_from_state(state: OrchestratorState) -> list[str]:
    if state.response.metadata.get("route") != "image_generation":
        return []
    raw = state.response.metadata.get("image_urls")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(url).strip() for url in raw if isinstance(url, str) and str(url).strip()]


def final_answer_from_state(state: OrchestratorState) -> str:
    image_urls = image_urls_from_state(state)
    if image_urls:
        return "\n\n".join(
            f"![Generated image {index}]({url})"
            for index, url in enumerate(image_urls, start=1)
        )
    return state.response.final_response.strip() or FALLBACK_NO_ANSWER


def usage_from_response_state(state: OrchestratorState) -> OpenAIUsage:
    usage = state.response.usage
    return OpenAIUsage(
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        total_tokens=int(usage.get("total_tokens", 0) or 0),
    )


def orchestrator_chat_result(thread_id: str, state: OrchestratorState) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "answer": final_answer_from_state(state),
        "request": state.request.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"original_query", "resolved_query", "is_followup", "followup_confidence"},
        ),
        "execution": state.execution.model_dump(mode="json", exclude_none=True),
        "evidence": state.evidence.model_dump(mode="json", exclude_none=True),
        "response": state.response.model_dump(mode="json", exclude_none=True),
        "debug": state.debug.model_dump(mode="json", exclude_none=True),
        "used_models": state.debug.used_models,
        "used_tools": state.debug.used_tools,
        "metadata": state.response.metadata,
    }


def completion_from_state(
    *,
    request_id: str,
    payload: OpenAIChatCompletionRequest,
    state: OrchestratorState,
    thread_id: str,
) -> OpenAIChatCompletionResponse:
    response_metadata = dict(state.response.metadata)
    response_metadata.update({"thread_id": thread_id, "request_id": request_id})
    return OpenAIChatCompletionResponse(
        id=request_id,
        created=int(time()),
        model=str(payload.model),
        choices=[
            OpenAIChatCompletionChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=final_answer_from_state(state)),
                finish_reason="stop",
            )
        ],
        usage=usage_from_response_state(state),
        metadata=response_metadata,
    )


__all__ = [
    "completion_from_state",
    "final_answer_from_state",
    "graph_input",
    "input_state_from_request_state",
    "native_thread_id",
    "openai_request_from_chat_request",
    "openai_thread_id",
    "orchestrator_chat_result",
    "request_headers",
    "usage_from_response_state",
]
