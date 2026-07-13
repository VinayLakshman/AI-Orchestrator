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
    app.state.runtime = runtime

    # Keep the resident controller loaded before serving traffic.
    with suppress(Exception):
        await runtime.model_manager.warm_controller()

    cleanup_task = asyncio.create_task(_cleanup_streams(runtime))

    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        await runtime.ollama_client.client.aclose()  # type: ignore[union-attr]
        await runtime.knowledge_client.client.aclose()  # type: ignore[union-attr]


settings = get_settings()

app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(api_router)
