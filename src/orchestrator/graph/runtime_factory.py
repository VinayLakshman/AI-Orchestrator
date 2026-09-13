"""Application dependency construction for the LangGraph runtime."""

from __future__ import annotations

import httpx

from ..clients.knowledge import KnowledgeClient
from ..clients.llama_cpp import LlamaCppClient
from ..clients.registry import ClientRegistry
from ..clients.searxng import SearXNGClient
from ..controller.engine import ControllerEngine
from ..logging import get_logger
from ..models.manager import ModelManager
from ..runtime.inference_gateway import ManagedInferenceClient
from ..runtime.model_provider import ModelProvider
from ..settings import Settings
from ..streaming.hub import StreamHub
from ..vision.pipeline import VisionPipeline
from .build import OrchestratorRuntime, build_graph

logger = get_logger(__name__)


async def build_runtime(settings: Settings) -> OrchestratorRuntime:
    """Construct all application dependencies with explicit ownership."""
    logger.info("registering runtime dependencies")
    from ..clients.openwebui import OpenWebUIClient
    from ..runtime.legacy_docker import DockerRuntime
    from ..runtime.lifecycle import ModelLifecycle

    provider = ModelProvider(settings)
    docker = DockerRuntime()

    router_health_timeout = max(1.0, min(5.0, float(settings.health_timeout_s)))
    async with httpx.AsyncClient(
        base_url=provider.router_origin_url,
        timeout=httpx.Timeout(router_health_timeout),
        follow_redirects=False,
    ) as health_client:
        try:
            response = await health_client.get("/health")
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Router health check failed at {provider.router_origin_url}/health"
            ) from exc

    knowledge_http = httpx.AsyncClient(
        base_url=settings.knowledge_service_url,
        timeout=settings.request_timeout_s,
    )
    client_registry = ClientRegistry()

    if settings.openwebui_base_url:
        openwebui_http = httpx.AsyncClient(
            base_url=settings.openwebui_base_url,
            timeout=max(settings.image_generation_timeout, 60.0),
        )
        client_registry.register(
            "openwebui",
            OpenWebUIClient(settings=settings, client=openwebui_http),
        )

    router_http = httpx.AsyncClient(
        base_url=provider.router_base_url,
        timeout=httpx.Timeout(settings.request_timeout_s),
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        headers={
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {settings.llama_cpp_api_key}"}
                if settings.llama_cpp_api_key
                else {}
            ),
        },
    )
    router_client = LlamaCppClient(
        settings=settings,
        client=router_http,
        base_url=provider.router_base_url,
    )
    for role in ("controller", "reasoning", "coder", "vision"):
        client_registry.register(role, router_client)

    knowledge_client = KnowledgeClient(settings=settings, client=knowledge_http)
    searxng_client: SearXNGClient | None = None
    if settings.web_search_enabled:
        web_http = httpx.AsyncClient(
            base_url=settings.web_search_url,
            timeout=settings.web_search_timeout_s,
        )
        searxng_client = SearXNGClient(settings=settings, client=web_http)

    model_manager = ModelManager(
        settings=settings,
        client_registry=client_registry,
        provider=provider,
    )
    controller = ControllerEngine(settings=settings, models=model_manager)
    vision_fetch_http = httpx.AsyncClient(
        base_url=settings.vision_fetch_base_url,
        timeout=settings.vision_timeout_s,
    )
    stream_hub = StreamHub(max_events=settings.stream_replay_max_events)

    model_lifecycle = ModelLifecycle(
        settings=settings,
        models=model_manager,
        docker=docker,
    )
    for role in ("controller", "reasoning", "coder", "vision"):
        client_registry.register(
            role,
            ManagedInferenceClient(
                role=role,
                transport=router_client,
                lifecycle=model_lifecycle,
                model_name=provider.model_for_role(role),
            ),
        )

    vision_pipeline = VisionPipeline(
        settings=settings,
        client=vision_fetch_http,
        model_client=client_registry.get("vision"),
    )
    model_lifecycle.start_background_cleanup()

    graph, checkpointer = build_graph(
        settings=settings,
        controller=controller,
        knowledge_client=knowledge_client,
        client_registry=client_registry,
        model_lifecycle=model_lifecycle,
        vision_pipeline=vision_pipeline,
        searxng_client=searxng_client,
    )
    runtime = OrchestratorRuntime(
        settings=settings,
        model_manager=model_manager,
        controller=controller,
        knowledge_client=knowledge_client,
        model_lifecycle=model_lifecycle,
        client_registry=client_registry,
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


__all__ = ["build_runtime"]
