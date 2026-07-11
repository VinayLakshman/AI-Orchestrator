from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..clients import KnowledgeClient, OllamaClient
from ..router import RequestRouter
from ..router.classifier import RoutingClassifier
from ..settings import Settings
from ..vision import VisionPipeline
from .nodes import make_clarify_node, make_generate_node, make_retrieve_node, make_route_node, make_vision_node
from .state import OrchestratorState


CheckpointerKind = Literal["memory", "sqlite"]


@dataclass(slots=True)
class OrchestratorRuntime:
    settings: Settings
    router: RequestRouter
    knowledge_client: KnowledgeClient
    ollama_client: OllamaClient
    vision_pipeline: VisionPipeline
    graph: Any
    checkpointer: Any


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
    router: RequestRouter,
    knowledge_client: KnowledgeClient,
    ollama_client: OllamaClient,
    vision_pipeline: VisionPipeline,
) -> tuple[Any, Any]:
    builder = StateGraph(OrchestratorState)

    vision_node = make_vision_node(vision_pipeline, settings)
    route_node = make_route_node(router, settings)
    retrieve_node = make_retrieve_node(knowledge_client, settings)
    generate_node = make_generate_node(ollama_client, settings)
    clarify_node = make_clarify_node()

    builder.add_node("vision", vision_node)
    builder.add_node("route", route_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("clarify", clarify_node)

    builder.add_edge(START, "vision")
    builder.add_edge("vision", "route")

    def select_next(state: OrchestratorState) -> str:
        route_raw = state.get("route") or {}
        route_name = str(route_raw.get("route", state.get("route_name", "general")))

        if route_name == "clarify":
            return "clarify"

        if route_name in {"rag", "multi_step"}:
            route_dict = route_raw if isinstance(route_raw, dict) else {}
            if route_dict.get("needs_rag", False):
                return "retrieve"
            return "generate"

        return "generate"

    builder.add_conditional_edges(
        "route",
        select_next,
        {
            "retrieve": "retrieve",
            "generate": "generate",
            "clarify": "clarify",
        },
    )

    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("clarify", END)

    checkpointer, _kind = build_checkpointer(settings)
    graph = builder.compile(checkpointer=checkpointer)

    return graph, checkpointer


async def build_runtime(settings: Settings) -> OrchestratorRuntime:
    ollama_http = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=settings.request_timeout_s)
    knowledge_http = httpx.AsyncClient(
        base_url=settings.knowledge_service_url,
        timeout=settings.request_timeout_s,
    )

    classifier = RoutingClassifier(settings=settings, client=ollama_http)
    router = RequestRouter(settings=settings, classifier=classifier)
    ollama_client = OllamaClient(settings=settings, client=ollama_http)
    knowledge_client = KnowledgeClient(settings=settings, client=knowledge_http)
    vision_pipeline = VisionPipeline(settings=settings, client=ollama_http)

    graph, checkpointer = build_graph(
        settings=settings,
        router=router,
        knowledge_client=knowledge_client,
        ollama_client=ollama_client,
        vision_pipeline=vision_pipeline,
    )

    return OrchestratorRuntime(
        settings=settings,
        router=router,
        knowledge_client=knowledge_client,
        ollama_client=ollama_client,
        vision_pipeline=vision_pipeline,
        graph=graph,
        checkpointer=checkpointer,
    )