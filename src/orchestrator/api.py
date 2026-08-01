from __future__ import annotations

import asyncio
from contextlib import suppress
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from orchestrator.logging import get_logger
from orchestrator.logging.request_summary import log_request_summary

from .common.constants import THREAD_ID_MAX_LENGTH
from .graph.build import OrchestratorRuntime
from .models.chat import ChatRequest
from .models.state import OrchestratorState, RequestState
from .request_normalizer import normalize_openai_request
from .preprocessing.conversation_resolver import resolve_conversation_context
from .schemas import (
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIMessage,
    OpenAIModelCard,
    OpenAIModelListResponse,
    OpenAIUsage,
)
from .streaming.context import stream_scope
from .streaming.models import StreamKind
from .streaming.publisher import StreamPublisher
from .streaming.sse import openai_chunk, openai_done

router = APIRouter(tags=["orchestrator"])
logger = get_logger(__name__)


def get_runtime(request: Request) -> OrchestratorRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Orchestrator runtime is not ready")
    return runtime


def _request_headers(request_id: str, thread_id: str) -> dict[str, str]:
    return {
        "cache-control": "no-cache",
        "connection": "keep-alive",
        "x-accel-buffering": "no",
        "x-orchestrator-request-id": request_id,
        "x-orchestrator-thread-id": thread_id,
    }


def _thread_id_from_request(payload: ChatRequest) -> str:
    if payload.thread_id:
        return payload.thread_id[:THREAD_ID_MAX_LENGTH]
    return str(uuid4())


def _openai_request_from_chat_request(payload: ChatRequest) -> OpenAIChatCompletionRequest:
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
        metadata=payload.metadata,
    )


def _input_state_from_request_state(
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
                "request_headers": _request_headers(request_id, thread_id),
            },
        }
    )

    return OrchestratorState(request=request_state)


def _final_answer_from_state(state: OrchestratorState) -> str:
    answer = state.response.final_response.strip()
    return answer or "I could not generate a complete answer for that request. Please try again with a little more detail."


def _usage_from_response_state(state: OrchestratorState) -> OpenAIUsage:
    usage = state.response.usage
    return OpenAIUsage(
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        total_tokens=int(usage.get("total_tokens", 0) or 0),
    )


def _orchestrator_chat_result(thread_id: str, state: OrchestratorState) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "answer": _final_answer_from_state(state),
        "request": state.request.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "original_query",
                "resolved_query",
                "is_followup",
                "followup_confidence",
            },
        ),
        "execution": state.execution.model_dump(mode="json", exclude_none=True),
        "evidence": state.evidence.model_dump(mode="json", exclude_none=True),
        "response": state.response.model_dump(mode="json", exclude_none=True),
        "debug": state.debug.model_dump(mode="json", exclude_none=True),
        "used_models": state.debug.used_models,
        "used_tools": state.debug.used_tools,
        "metadata": state.response.metadata,
    }


def _completion_from_state(
    *,
    request_id: str,
    payload: OpenAIChatCompletionRequest,
    state: OrchestratorState,
    thread_id: str,
) -> OpenAIChatCompletionResponse:
    answer = _final_answer_from_state(state)
    usage = _usage_from_response_state(state)
    response_metadata = dict(state.response.metadata)
    response_metadata.update(
        {
            "thread_id": thread_id,
            "request_id": request_id,
        }
    )

    return OpenAIChatCompletionResponse(
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
        usage=usage,
        metadata=response_metadata,
    )


async def _run_graph_with_stream(
    *,
    runtime: OrchestratorRuntime,
    request_id: str,
    thread_id: str,
    state_input: OrchestratorState,
    publisher: StreamPublisher,
) -> OrchestratorState:
    logger.debug("GRAPH: entered _run_graph_with_stream")
    try:
        async with stream_scope(publisher):
            await publisher.graph_started()
            logger.debug("GRAPH: invoking graph")
            result = await runtime.graph.ainvoke(
                state_input,
                config={"configurable": {"thread_id": thread_id}},
            )
            logger.debug("GRAPH: graph finished")
            route_name = result.execution.plan.route.value
            await publisher.graph_finished(route=route_name)
            return result
    except Exception as exc:
        await publisher.graph_failed(str(exc))
        raise
    finally:
        logger.debug("GRAPH: closing stream")
        await runtime.stream_hub.close(request_id)


def _emit_request_summary(
    *,
    request_id: str,
    state: OrchestratorState,
    execution_trace: list[dict[str, Any]] | None,
    total_duration_ms: int | float,
) -> None:
    log_request_summary(
        request_id=request_id,
        state=state,
        execution_trace=execution_trace,
        timings=state.debug.timings,
        total_duration_ms=total_duration_ms,
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


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    runtime: OrchestratorRuntime = Depends(get_runtime),
) -> dict[str, Any]:
    thread_id = _thread_id_from_request(payload)
    state_input = await _input_state_from_request(payload, runtime=runtime, thread_id=thread_id)
    request_id = state_input.request.request_id
    started_at = perf_counter()

    result: OrchestratorState = await runtime.graph.ainvoke(
        state_input,
        config={"configurable": {"thread_id": thread_id}},
    )

    _emit_request_summary(
        request_id=request_id,
        state=result,
        execution_trace=result.debug.execution_trace,
        total_duration_ms=(perf_counter() - started_at) * 1000.0,
    )

    return _orchestrator_chat_result(thread_id, result)


async def _input_state_from_request(
    payload: ChatRequest,
    *,
    runtime: OrchestratorRuntime,
    request_id: str | None = None,
    thread_id: str | None = None,
) -> OrchestratorState:
    request_state = normalize_openai_request(_openai_request_from_chat_request(payload))
    resolved = await resolve_conversation_context(
        request_state,
        settings=runtime.settings,
        model_manager=runtime.model_manager,
        ollama_client=runtime.ollama_client,
    )
    return _input_state_from_request_state(
        resolved.request,
        thread_id=thread_id or _thread_id_from_request(payload),
        request_id=request_id,
        model=payload.model or "orchestrator",
        stream=payload.stream,
    )


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
        headers=_request_headers(request_id, stream.conversation_id or ""),
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
    request_state = normalize_openai_request(payload)
    resolved = await resolve_conversation_context(
        request_state,
        settings=runtime.settings,
        model_manager=runtime.model_manager,
        ollama_client=runtime.ollama_client,
    )
    thread_id = str(uuid4())
    started_at = perf_counter()

    state_input = _input_state_from_request_state(
        resolved.request,
        thread_id=thread_id,
        request_id=request_id,
        model=str(payload.model or "orchestrator"),
        stream=payload.stream,
    )

    stream = runtime.stream_hub.get_or_create(request_id, conversation_id=thread_id)
    publisher = StreamPublisher(stream)

    if payload.stream:

        async def sse_generator():
            token_seen = False
            result: OrchestratorState | None = None
            graph_task: asyncio.Task | None = None
            event_queue: asyncio.Queue[Any] = asyncio.Queue()
            next_heartbeat = asyncio.get_running_loop().time() + 10

            logger.debug("SSE: generator started")

            yield openai_chunk(
                request_id=request_id,
                model=str(payload.model),
                role="assistant",
            )

            async def relay_events() -> None:
                try:
                    async for event in stream.subscribe(after_seq=0):
                        await event_queue.put(event)
                finally:
                    await event_queue.put(None)

            try:
                logger.debug("SSE: creating graph task")
                graph_task = asyncio.create_task(
                    _run_graph_with_stream(
                        runtime=runtime,
                        request_id=request_id,
                        thread_id=thread_id,
                        state_input=state_input,
                        publisher=publisher,
                    ),
                    name=f"orchestrator-stream-{request_id}",
                )

                def _graph_done(task: asyncio.Task):
                    try:
                        exc = task.exception()
                        if exc:
                            logger.exception("GRAPH TASK FAILED", exc_info=exc)
                        else:
                            logger.debug("GRAPH TASK COMPLETED")
                    except asyncio.CancelledError:
                        logger.debug("GRAPH TASK CANCELLED")

                graph_task.add_done_callback(_graph_done)
                logger.debug("SSE: graph task created")

                relay_task = asyncio.create_task(
                    relay_events(),
                    name=f"orchestrator-events-{request_id}",
                )
                while True:
                    timeout = max(0, next_heartbeat - asyncio.get_running_loop().time())
                    if timeout == 0:
                        yield ": keep-alive\n\n"
                        next_heartbeat = asyncio.get_running_loop().time() + 10
                        continue
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        next_heartbeat = asyncio.get_running_loop().time() + 10
                        continue

                    if event is None:
                        break

                    logger.debug(
                        "SSE: event kind=%s payload=%s",
                        event.kind,
                        event.payload,
                    )
                    if event.kind != StreamKind.LLM_TOKEN:
                        continue

                    payload_data = event.payload or {}
                    token = str(
                        payload_data.get("token")
                        or payload_data.get("content")
                        or payload_data.get("text")
                        or ""
                    )
                    if not token:
                        continue

                    token_seen = True

                    yield openai_chunk(
                        request_id=request_id,
                        model=str(payload.model),
                        content=token,
                    )
                logger.debug("SSE: subscription finished")

                with suppress(Exception):
                    result = await graph_task

                if not token_seen:
                    answer = _final_answer_from_state(result) if result is not None else ""
                    if answer:
                        yield openai_chunk(
                            request_id=request_id,
                            model=str(payload.model),
                            content=answer,
                        )

                if result is not None:
                    _emit_request_summary(
                        request_id=request_id,
                        state=result,
                        execution_trace=result.debug.execution_trace,
                        total_duration_ms=(perf_counter() - started_at) * 1000.0,
                    )

                yield openai_chunk(
                    request_id=request_id,
                    model=str(payload.model),
                    finish_reason="stop",
                )
                yield openai_done()

            except asyncio.CancelledError:
                if graph_task is not None:
                    graph_task.cancel()
                raise
            finally:
                if "relay_task" in locals() and not relay_task.done():
                    relay_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await relay_task
                if graph_task is not None and not graph_task.done():
                    graph_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await graph_task

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers=_request_headers(request_id, thread_id),
        )

    result: OrchestratorState = await runtime.graph.ainvoke(
        state_input,
        config={"configurable": {"thread_id": thread_id}},
    )

    _emit_request_summary(
        request_id=request_id,
        state=result,
        execution_trace=result.debug.execution_trace,
        total_duration_ms=(perf_counter() - started_at) * 1000.0,
    )

    completion = _completion_from_state(
        request_id=request_id,
        payload=payload,
        state=result,
        thread_id=thread_id,
    )

    await stream.close()
    return completion
