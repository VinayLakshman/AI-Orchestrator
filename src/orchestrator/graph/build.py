from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..clients.knowledge import KnowledgeClient
from ..clients.searxng import SearXNGClient
from ..clients.ollama import OllamaClient
from ..controller.engine import ControllerEngine
from ..models.manager import ModelManager
from ..models.state import OrchestratorState
from ..logging import get_logger
from ..settings import Settings
from ..specialists.web import WebSpecialist
from ..streaming.hub import StreamHub
from ..vision.pipeline import VisionPipeline
from .nodes import (
    make_clarify_node,
    make_controller_plan_node,
    make_controller_validate_node,
    make_coder_node,
    make_finalize_node,
    make_knowledge_node,
    make_web_node,
    make_prepare_node,
    make_reasoning_node,
    make_tools_node,
    make_vision_node,
    _state_snapshot,
    _log_transition,
)
from .instrumentation import timed_node


CheckpointerKind = Literal["memory", "sqlite"]
logger = get_logger(__name__)


@dataclass(slots=True)
class TypedGraphFacade:
    graph: Any

    def __getattr__(self, name: str) -> Any:
        return getattr(self.graph, name)

    @staticmethod
    def _unwrap_state(result: Any) -> OrchestratorState:
        state = getattr(result, "value", result)

        if isinstance(state, OrchestratorState):
            return state

        if isinstance(state, dict):
            return OrchestratorState.model_validate(state)

        raise TypeError(
            f"Expected OrchestratorState or dict from graph, got {type(state).__name__}"
        )

    async def ainvoke(self, *args: Any, **kwargs: Any) -> OrchestratorState:
        kwargs.setdefault("version", "v2")
        result = await self.graph.ainvoke(*args, **kwargs)
        return self._unwrap_state(result)

    def invoke(self, *args: Any, **kwargs: Any) -> OrchestratorState:
        kwargs.setdefault("version", "v2")
        result = self.graph.invoke(*args, **kwargs)
        return self._unwrap_state(result)

    async def astream(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("version", "v2")
        async for chunk in self.graph.astream(*args, **kwargs):
            yield self._normalize_stream_chunk(chunk, stream_mode=kwargs.get("stream_mode", "values"))

    def stream(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("version", "v2")
        for chunk in self.graph.stream(*args, **kwargs):
            yield self._normalize_stream_chunk(chunk, stream_mode=kwargs.get("stream_mode", "values"))

    @staticmethod
    def _normalize_stream_chunk(chunk: Any, *, stream_mode: str) -> Any:
        if stream_mode != "values":
            return chunk
        if isinstance(chunk, OrchestratorState):
            return chunk
        if hasattr(chunk, "data") and isinstance(chunk.data, OrchestratorState):
            return chunk.data
        if isinstance(chunk, tuple) and chunk and isinstance(chunk[-1], OrchestratorState):
            return chunk[-1]
        if isinstance(chunk, dict):
            if "data" in chunk:
                data = chunk["data"]

                if isinstance(data, dict):
                    return OrchestratorState.model_validate(data)

                if isinstance(data, OrchestratorState):
                    return data

            return OrchestratorState.model_validate(chunk)
        return chunk


@dataclass(slots=True)
class OrchestratorRuntime:
    settings: Settings
    model_manager: ModelManager
    controller: ControllerEngine
    knowledge_client: KnowledgeClient
    ollama_client: OllamaClient
    vision_pipeline: VisionPipeline
    stream_hub: StreamHub
    graph: Any
    checkpointer: Any
    searxng_client: SearXNGClient | None = None

    def validate_dependencies(self) -> None:
        required = {
            "settings": self.settings,
            "model_manager": self.model_manager,
            "controller": self.controller,
            "knowledge_client": self.knowledge_client,
            "ollama_client": self.ollama_client,
            "vision_pipeline": self.vision_pipeline,
            "stream_hub": self.stream_hub,
            "graph": self.graph,
            "checkpointer": self.checkpointer,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"Runtime dependency registration incomplete: {', '.join(missing)}")
        if self.settings.web_search_enabled and self.searxng_client is None:
            raise RuntimeError(
                "WEB_SEARCH_ENABLED is true but the SearXNG client was not registered"
            )

    async def close(self) -> None:
        """Close runtime-owned transports exactly once."""
        clients = [self.ollama_client.client, self.knowledge_client.client]
        if self.searxng_client is not None:
            clients.append(self.searxng_client.client)
        closed: set[int] = set()
        for client in clients:
            if client is None or id(client) in closed:
                continue
            closed.add(id(client))
            await client.aclose()


def build_checkpointer(settings: Settings) -> tuple[Any, CheckpointerKind]:
    sqlite_path = settings.checkpoint_sqlite_path.strip()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

        if sqlite_path.startswith("sqlite:///"):
            conn_string = sqlite_path
        elif sqlite_path.startswith("sqlite:"):
            conn_string = sqlite_path
        else:
            conn_string = f"sqlite:///{sqlite_path}"

        return SqliteSaver.from_conn_string(conn_string), "sqlite"
    except Exception:
        return MemorySaver(), "memory"


def build_graph(
    settings: Settings,
    controller: ControllerEngine,
    knowledge_client: KnowledgeClient,
    ollama_client: OllamaClient,
    vision_pipeline: VisionPipeline,
    searxng_client: SearXNGClient | None = None,
) -> tuple[Any, Any]:
    builder = StateGraph(
        OrchestratorState,
        input_schema=OrchestratorState,
        output_schema=OrchestratorState,
    )

    prepare_node = make_prepare_node(settings)
    plan_node = make_controller_plan_node(controller, settings)
    vision_node = make_vision_node(vision_pipeline, settings)
    knowledge_node = make_knowledge_node(knowledge_client, settings)
    web_node = make_web_node(WebSpecialist(searxng_client), settings)
    coder_node = make_coder_node(controller, settings)
    tools_node = make_tools_node(settings)
    validate_node = make_controller_validate_node(controller, settings)
    reasoning_node = make_reasoning_node(controller, settings)
    clarify_node = make_clarify_node()
    finalize_node = make_finalize_node(controller, settings)

    prepare_node = timed_node("prepare", prepare_node, display_name="Prepare")
    plan_node = timed_node("planner", plan_node, display_name="Planner")
    vision_node = timed_node("vision", vision_node, display_name="Vision")
    knowledge_node = timed_node("knowledge", knowledge_node, display_name="Knowledge")
    web_node = timed_node("web", web_node, display_name="Web")
    coder_node = timed_node("coder", coder_node, display_name="Code")
    tools_node = timed_node("tools", tools_node, display_name="Tools")
    validate_node = timed_node("validation", validate_node, display_name="Validation")
    reasoning_node = timed_node("reasoning", reasoning_node, display_name="Reasoning")
    clarify_node = timed_node("clarify", clarify_node, display_name="Clarification")
    finalize_node = timed_node("finalize", finalize_node, display_name="Finalizer")

    builder.add_node("prepare", prepare_node)
    builder.add_node("plan", plan_node)
    builder.add_node("vision", vision_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("web", web_node)
    builder.add_node("coder", coder_node)
    builder.add_node("tools", tools_node)
    builder.add_node("validate", validate_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "plan")

    def _next_node(state: OrchestratorState) -> str:
        # DEBUG: trace runtime queue -> selected node
        runtime = state.execution.runtime
        logger.debug(
            "DEBUG graph_next_node queue=%s current_index=%s current_specialist=%s plan_classification=%s validation_action=%s validation_complete=%s requires_reasoning=%s requires_clarification=%s",
            [s.value for s in (runtime.queue or [])],
            runtime.current_index,
            runtime.current_specialist.value if runtime.current_specialist else None,
            getattr(state.execution.plan, "classification", None),
            state.execution.validation.action.value if state.execution.validation else None,
            state.execution.validation.complete if state.execution.validation else None,
            state.execution.validation.requires_reasoning if state.execution.validation else None,
            state.execution.validation.requires_clarification if state.execution.validation else None,
        )
        validation = state.execution.validation


        if validation is not None:
            if validation.retry:
                current = state.execution.runtime.current_specialist
                if current is not None:
                    return current.value

            if validation.requires_reasoning:
                return "reasoning"

            if validation.requires_clarification:
                return "clarify"

            if validation.complete:
                return "finalize"

        runtime = state.execution.runtime
        queue = runtime.queue

        if runtime.current_index >= len(queue):
            return "finalize"

        specialist = queue[runtime.current_index]
        return specialist.value


    def route_after_plan(state: OrchestratorState) -> str:
        selected = _next_node(state)

        _log_transition(
            "route_after_plan",
            selected_next_node=selected,
            **_state_snapshot(state),
        )

        return selected


    def route_after_validate(state: OrchestratorState) -> str:
        selected = _next_node(state)

        _log_transition(
            "route_after_validate",
            selected_next_node=selected,
            **_state_snapshot(state),
        )

        return selected

    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "vision": "vision",
            "knowledge": "knowledge",
            "web": "web",
            "coder": "coder",
            "tools": "tools",
            "reasoning": "reasoning",
            "finalize": "finalize",
            "clarify": "clarify",
        },
    )

    builder.add_edge("vision", "validate")
    builder.add_edge("knowledge", "validate")
    builder.add_edge("web", "validate")
    builder.add_edge("coder", "validate")
    builder.add_edge("tools", "validate")

    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "vision": "vision",
            "knowledge": "knowledge",
            "web": "web",
            "coder": "coder",
            "tools": "tools",
            "reasoning": "reasoning",
            "finalize": "finalize",
            "clarify": "clarify",
        },
    )

    builder.add_edge("reasoning", "finalize")
    builder.add_edge("clarify", END)
    builder.add_edge("finalize", END)

    checkpointer, _kind = build_checkpointer(settings)
    graph = TypedGraphFacade(builder.compile(checkpointer=checkpointer))

    return graph, checkpointer


async def build_runtime(settings: Settings) -> OrchestratorRuntime:
    logger.info("registering runtime dependencies")
    ollama_http = httpx.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=settings.request_timeout_s,
    )
    knowledge_http = httpx.AsyncClient(
        base_url=settings.knowledge_service_url,
        timeout=settings.request_timeout_s,
    )

    ollama_client = OllamaClient(settings=settings, client=ollama_http)
    knowledge_client = KnowledgeClient(settings=settings, client=knowledge_http)
    searxng_client: SearXNGClient | None = None
    if settings.web_search_enabled:
        web_http = httpx.AsyncClient(
            base_url=settings.web_search_url,
            timeout=settings.web_search_timeout_s,
        )
        searxng_client = SearXNGClient(settings=settings, client=web_http)
    model_manager = ModelManager(settings=settings, ollama_client=ollama_client)
    controller = ControllerEngine(settings=settings, ollama=ollama_client, models=model_manager)
    vision_pipeline = VisionPipeline(settings=settings, client=ollama_http)
    stream_hub = StreamHub()

    graph, checkpointer = build_graph(
        settings=settings,
        controller=controller,
        knowledge_client=knowledge_client,
        searxng_client=searxng_client,
        ollama_client=ollama_client,
        vision_pipeline=vision_pipeline,
    )

    runtime = OrchestratorRuntime(
        settings=settings,
        model_manager=model_manager,
        controller=controller,
        knowledge_client=knowledge_client,
        ollama_client=ollama_client,
        vision_pipeline=vision_pipeline,
        stream_hub=stream_hub,
        graph=graph,
        checkpointer=checkpointer,
        searxng_client=searxng_client,
    )
    runtime.validate_dependencies()
    logger.info(
        "runtime dependency registration complete web_search=%s knowledge=%s vision=%s",
        runtime.searxng_client is not None,
        settings.enable_rag,
        settings.enable_vision,
    )
    return runtime
