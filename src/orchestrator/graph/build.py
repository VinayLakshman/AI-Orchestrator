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


CheckpointerKind = Literal["memory", "sqlite"]
logger = get_logger(__name__)


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
    builder = StateGraph(OrchestratorState)

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
    graph = builder.compile(checkpointer=checkpointer)

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
