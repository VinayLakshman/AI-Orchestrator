from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .graph import OrchestratorRuntime
from .schemas import ChatMessage, ChatRequest, OrchestratorResponse
from .settings import get_settings

router = APIRouter(tags=["orchestrator"])


def get_runtime(request: Request) -> OrchestratorRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Orchestrator runtime is not ready")
    return runtime


class OpenAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key in ("authorization", "cookie"):
        value = request.headers.get(key)
        if value:
            headers[key] = value
    return headers


def _to_chat_request(payload: OpenAIChatRequest, thread_id: str | None = None) -> ChatRequest:
    messages = [ChatMessage.model_validate(item) for item in payload.messages]
    return ChatRequest(
        messages=messages,
        thread_id=thread_id,
        model=payload.model,
        stream=payload.stream,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        metadata=payload.metadata,
    )


def _thread_id_from_request(payload: ChatRequest) -> str:
    if payload.thread_id:
        return payload.thread_id[:255]
    return str(uuid4())


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
        "knowledge": [],
    }


def _response_from_state(thread_id: str, state: dict[str, Any]) -> OrchestratorResponse:
    route = state.get("route") or {}

    return OrchestratorResponse(
        thread_id=thread_id,
        route=route,
        answer=state.get("answer") or "",
        used_models=state.get("used_models") or [],
        used_tools=state.get("used_tools") or [],
        knowledge=state.get("knowledge") or [],
        vision=state.get("vision"),
        vision_context=state.get("vision_context") or "",
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


@router.post("/v1/chat/completions")
async def openai_chat_completions(
    request: Request,
    runtime: OrchestratorRuntime = Depends(get_runtime),
) -> JSONResponse:
    body = await request.json()
    payload = _to_chat_request(OpenAIChatRequest.model_validate(body))
    thread_id = _thread_id_from_request(payload)
    state_input = _input_state_from_request(payload, request)

    result = await runtime.graph.ainvoke(
        state_input,
        config={"configurable": {"thread_id": thread_id}},
    )

    answer = result.get("answer", "")
    model_used = (result.get("used_models") or [payload.model or "orchestrator"])[-1]

    response = {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "metadata": {
            "thread_id": thread_id,
            "route": result.get("route", {}),
            "used_models": result.get("used_models", []),
            "used_tools": result.get("used_tools", []),
            "vision": result.get("vision"),
            "vision_context": result.get("vision_context", ""),
        },
    }

    return JSONResponse(response)