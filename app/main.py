from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.request_id import RequestIDMiddleware
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.core.logging import configure_logging
from app.settings import get_settings

configure_logging()
settings = get_settings()
cors_origins = [origin.strip() for origin in settings.cors_allow_origins.split(',') if origin.strip()]

app = FastAPI(title="AI Orchestrator", version="0.1.0")
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(metrics_router)
