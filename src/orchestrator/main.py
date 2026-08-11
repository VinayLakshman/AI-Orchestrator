from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI

from .api import router as api_router
from .graph.build import build_runtime
from .logging import get_logger
from .logging import configure_logging
from .settings import get_settings


logger = get_logger(__name__)


async def _cleanup_streams(runtime) -> None:
    while True:
        await asyncio.sleep(300)
        await runtime.stream_hub.cleanup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "startup settings loaded: log_level=%s cwd=%s env_file=%s env_exists=%s",
        settings.log_level,
        Path.cwd(),
        ".env",
        Path(".env").exists(),
    )

    runtime = await build_runtime(settings)
    enabled = ["coder", "reasoning"]
    disabled: list[str] = []
    if settings.mcp_enabled:
        enabled.append("tools")
    else:
        disabled.append("tools")
    if settings.enable_rag:
        enabled.append("knowledge")
    else:
        disabled.append("knowledge")
    if settings.enable_vision:
        enabled.append("vision")
    else:
        disabled.append("vision")
    if settings.web_search_enabled:
        enabled.append("web")
    else:
        disabled.append("web")
    logger.info("runtime dependencies registered enabled_specialists=%s disabled_specialists=%s", enabled, disabled)
    logger.debug(
        "runtime dependency map settings=%s model_manager=%s controller=%s knowledge_client=%s searxng_client=%s client_registry=%s vision_pipeline=%s stream_hub=%s graph=%s checkpointer=%s",
        type(runtime.settings).__name__,
        type(runtime.model_manager).__name__,
        type(runtime.controller).__name__,
        type(runtime.knowledge_client).__name__,
        type(runtime.searxng_client).__name__ if runtime.searxng_client else None,
        type(runtime.client_registry).__name__,
        type(runtime.vision_pipeline).__name__,
        type(runtime.stream_hub).__name__,
        type(runtime.graph).__name__,
        type(runtime.checkpointer).__name__,
    )
    app.state.runtime = runtime

    cleanup_task = asyncio.create_task(_cleanup_streams(runtime))

    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        await runtime.close()


settings = get_settings()

app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(api_router)
