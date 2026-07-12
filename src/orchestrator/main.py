from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from .api import router as api_router
from .graph import build_runtime
from .logging import configure_logging
from .settings import get_settings


async def _cleanup_streams(runtime) -> None:
    while True:
        await asyncio.sleep(300)
        await runtime.stream_hub.cleanup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)

    runtime = await build_runtime(settings)
    app.state.runtime = runtime
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