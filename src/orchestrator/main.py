from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from .api import ZionRuntime, router
from .clients.knowledge import KnowledgeClient
from .clients.llama_cpp import LlamaCppClient
from .logging import configure_logging, get_logger
from .settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings)
    if settings.require_auth and not settings.api_key:
        raise RuntimeError("ZION_API_KEY must be configured while ZION_REQUIRE_AUTH is enabled")
    knowledge_http = httpx.AsyncClient(
        base_url=settings.knowledge_service_url.rstrip("/"),
        timeout=httpx.Timeout(settings.knowledge_timeout_s),
        headers=(
            {"Authorization": f"Bearer {settings.knowledge_service_api_key}"}
            if settings.knowledge_service_api_key
            else {}
        ),
    )
    llm_http = httpx.AsyncClient(
        base_url=settings.llm_base_url.rstrip("/"),
        timeout=httpx.Timeout(settings.llm_timeout_s),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}),
        },
    )
    knowledge = KnowledgeClient(settings, client=knowledge_http)
    llm = LlamaCppClient(settings, client=llm_http, base_url=settings.llm_base_url.rstrip("/"))
    app.state.settings = settings
    app.state.runtime = ZionRuntime(settings, knowledge, llm)
    logger.info("Zion RAG API started model=%s knowledge_service=%s", settings.llm_model,
                settings.knowledge_service_url)
    try:
        yield
    finally:
        await app.state.runtime.close()


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.include_router(router)
