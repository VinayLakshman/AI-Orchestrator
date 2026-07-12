from __future__ import annotations

from time import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .graph import OrchestratorRuntime
from .schemas import (
    ChatMessage,
    ChatRequest,
    KnowledgeRetrieveResponse,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIMessage,
    OpenAIModelCard,
    OpenAIModelListResponse,
    OrchestratorResponse,
    RouteDecision,
)
from .settings import get_settings

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
        return payload.thread_id[:255]
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


def _input_state_from_request(payload: ChatRequest, request: Request) -> dict[str, Any]:
    return {
        "thread_id": _thread_id_from_request(payload),
        "messages": [message.model_dump(exclude_none=True) for message in payload.messages],
        "metadata": {
            **payload.metadata,
            "requested_model": payload.model,
            "requested_stream": payload.stream,
            "temperature": payload.temperature,
            "max_tokens": payload.max_tokens,
            "request_headers": _request_headers(request),
        },
        "used_models": [],
        "used_tools": [],
        "knowledge_result": None,
    }


def _response_from_state(thread_id: str, state: dict[str, Any]) -> OrchestratorResponse:
    route = RouteDecision.model_validate(state.get("route") or {})
    knowledge_result = state.get("knowledge_result")

    if isinstance(knowledge_result, dict):
        knowledge_result = KnowledgeRetrieveResponse.model_validate(knowledge_result)

    return OrchestratorResponse(
        thread_id=thread_id,
        route=route,
        answer=state.get("answer", ""),
        used_models=state.get("used_models", []),
        used_tools=state.get("used_tools", []),
        knowledge_result=knowledge_result,
        vision=state.get("vision"),
        vision_context=state.get("vision_context", ""),
        metadata=state.get("metadata", {}),
    )


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
async def list_models() -> OpenAIModelListResponse:
    settings = get_settings()

    return OpenAIModelListResponse(
        data=[
            OpenAIModelCard(id="orchestrator", owned_by="local"),
            OpenAIModelCard(id=settings.general_model, owned_by="local"),
            OpenAIModelCard(id=settings.coder_model, owned_by="local"),
            OpenAIModelCard(id=settings.vision_model, owned_by="local"),
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


@router.post("/v1/chat/completions", response_model=OpenAIChatCompletionResponse)
async def openai_chat_completions(
    payload: OpenAIChatCompletionRequest,
    request: Request,
    runtime: OrchestratorRuntime = Depends(get_runtime),
) -> OpenAIChatCompletionResponse:
    chat_request = _to_chat_request(payload)
    thread_id = _thread_id_from_request(chat_request)
    state_input = _input_state_from_request(chat_request, request)

    result = await runtime.graph.ainvoke(
        state_input,
        config={"configurable": {"thread_id": thread_id}},
    )

    answer = result.get("answer", "")
    model_used = (result.get("used_models") or [payload.model])[ -1 ]

    return OpenAIChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex}",
        created=int(time()),
        model=model_used,
        choices=[
            OpenAIChatCompletionChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
        metadata={
            "thread_id": thread_id,
            "route": result.get("route", {}),
            "used_models": result.get("used_models", []),
            "used_tools": result.get("used_tools", []),
            "vision": result.get("vision"),
            "vision_context": result.get("vision_context", ""),
            "knowledge_result": result.get("knowledge_result"),
        },
    )