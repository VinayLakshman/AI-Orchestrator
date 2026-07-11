from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import router as api_router
from .graph import build_runtime
from .logging import configure_logging
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)

    runtime = await build_runtime(settings)
    app.state.runtime = runtime

    try:
        yield
    finally:
        await runtime.ollama_client.client.aclose()  # type: ignore[union-attr]
        await runtime.knowledge_client.client.aclose()  # type: ignore[union-attr]


settings = get_settings()

app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(api_router)