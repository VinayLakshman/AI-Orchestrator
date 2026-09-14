from __future__ import annotations

import asyncio
from contextlib import suppress
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from orchestrator.logging import get_logger
from orchestrator.logging.request_summary import log_request_summary

from .graph.build import OrchestratorRuntime
from .models.chat import ChatRequest
from .models.state import OrchestratorState, RequestState
from .request_normalizer import normalize_openai_request
from .preprocessing.conversation_resolver import resolve_conversation_context
from .schemas import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIModelCard,
    OpenAIModelListResponse,
)
from .http.adapters import (
    completion_from_state as _completion_from_state,
    final_answer_from_state as _final_answer_from_state,
    graph_input as _graph_input,
    input_state_from_request_state as _input_state_from_request_state,
    native_thread_id as _thread_id_from_request,
    openai_request_from_chat_request as _openai_request_from_chat_request,
    openai_thread_id as _openai_thread_id,
    orchestrator_chat_result as _orchestrator_chat_result,
    request_headers as _request_headers,
)
from .streaming.models import StreamKind
from .streaming.publisher import StreamPublisher
from .streaming.sse import openai_chunk, openai_done
from .runtime.metrics import runtime_metrics
from .http.streaming import run_graph_with_stream as _run_graph_with_stream

router = APIRouter(tags=["orchestrator"])
logger = get_logger(__name__)


def get_runtime(request: Request) -> OrchestratorRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Orchestrator runtime is not ready")
    return runtime


def _emit_request_summary(
    *,
    request_id: str,
    state: OrchestratorState,
    execution_trace: list[dict[str, Any]] | None,
    total_duration_ms: int | float,
) -> None:
    runtime_metrics.observe("orchestrator_total_latency_ms", float(total_duration_ms))
    memory = state.conversation.memory
    resolution = state.request.metadata.get("conversation_resolution") or {}
    runtime_metrics.observe(
        "orchestrator_conversation_memory_turns",
        float(len(memory.turns)),
    )
    memory_source = str(
        state.request.metadata.get("conversation_context_source") or "none"
    )
    runtime_metrics.increment(
        "orchestrator_conversation_memory_hit"
        if memory_source in {"checkpoint_memory", "request_history"}
        else "orchestrator_conversation_memory_miss",
        labels={"source": memory_source},
    )
    if isinstance(resolution, dict) and resolution.get("applied"):
        runtime_metrics.increment("orchestrator_followup_resolution_applied")
    for stage, duration_ms in state.debug.timings.items():
        runtime_metrics.observe(
            "orchestrator_stage_duration_ms",
            float(duration_ms),
            labels={"stage": stage},
        )
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


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(runtime_metrics.render(), media_type="text/plain; version=0.0.4")


@router.get("/v1/models", response_model=OpenAIModelListResponse)
async def list_models(runtime: OrchestratorRuntime = Depends(get_runtime)) -> OpenAIModelListResponse:
    return OpenAIModelListResponse(
        data=[
            OpenAIModelCard(id="orchestrator", owned_by="local"),
            OpenAIModelCard(id="controller", owned_by="local"),
            OpenAIModelCard(id="reasoning", owned_by="local"),
            OpenAIModelCard(id="coder", owned_by="local"),
            OpenAIModelCard(id="vision", owned_by="local"),
            OpenAIModelCard(id=runtime.settings.embedding_model, owned_by="local"),
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
        _graph_input(state_input),
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
    request_state = normalize_openai_request(
        _openai_request_from_chat_request(payload),
        request_id=request_id or "",
        thread_id=thread_id or "",
    )
    resolved_request = request_state
    # Legacy mode keeps the historical route-side resolver. Optimized mode
    # resolves inside the graph after checkpoint state has been restored.
    if runtime.settings.legacy_execution_mode:
        resolved_request = (
            await resolve_conversation_context(
                request_state,
                settings=runtime.settings,
                model_manager=runtime.model_manager,
                client_registry=runtime.client_registry,
            )
        ).request
    return _input_state_from_request_state(
        resolved_request,
        thread_id=thread_id or _thread_id_from_request(payload),
        request_id=request_id,
        model=payload.model or "orchestrator",
        stream=payload.stream and runtime.settings.enable_streaming,
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
    thread_id = _openai_thread_id(request, payload)
    request_state = normalize_openai_request(
        payload,
        request_id=request_id,
        thread_id=thread_id,
    )
    started_at = perf_counter()

    stream = runtime.stream_hub.get_or_create(request_id, conversation_id=thread_id)
    publisher = StreamPublisher(stream)

    if payload.stream and runtime.settings.enable_streaming:

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
            runtime_metrics.observe(
                "orchestrator_time_to_first_token_ms",
                (perf_counter() - started_at) * 1000.0,
            )

            # Defer model-backed conversation resolution until after the
            # initial SSE role chunk so clients receive an immediate response
            # signal even when the request has a long history.
            resolved_request = request_state
            if runtime.settings.legacy_execution_mode:
                resolved_request = (
                    await resolve_conversation_context(
                        request_state,
                        settings=runtime.settings,
                        model_manager=runtime.model_manager,
                        client_registry=runtime.client_registry,
                    )
                ).request
            state_input = _input_state_from_request_state(
                resolved_request,
                thread_id=thread_id,
                request_id=request_id,
                model=str(payload.model or "orchestrator"),
                stream=payload.stream and runtime.settings.enable_streaming,
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
                        "SSE: event kind=%s",
                        event.kind,
                    )

                    if event.kind != StreamKind.LLM_TOKEN:
                        # Lifecycle/state events (specialist progress,
                        # image_generation_started/progress/finished,
                        # validation, graph_finished/graph_failed, error)
                        # MUST reach the client in real time.
                        #
                        # Previously every non-token event was dropped here
                        # (`continue`), so the client saw no transition out of
                        # "planning" during the entire GPU-acquisition + image-
                        # generation window and had no signal separating image
                        # completion from graph completion.
                        #
                        # Relay them verbatim using the project's canonical
                        # StreamEvent SSE serialization (`id:`/`event:`/`data:`
                        # lines — same format as GET /v1/streams/{request_id}).
                        # Named SSE events are ignored by generic OpenAI chunk
                        # parsers, so this stays wire-compatible.
                        yield event.to_sse()
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

    resolved_request = request_state
    if runtime.settings.legacy_execution_mode:
        resolved_request = (
            await resolve_conversation_context(
                request_state,
                settings=runtime.settings,
                model_manager=runtime.model_manager,
                client_registry=runtime.client_registry,
            )
        ).request
    state_input = _input_state_from_request_state(
        resolved_request,
        thread_id=thread_id,
        request_id=request_id,
        model=str(payload.model or "orchestrator"),
        stream=payload.stream and runtime.settings.enable_streaming,
    )

    result: OrchestratorState = await runtime.graph.ainvoke(
        _graph_input(state_input),
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
