from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlamaCppModelConfig(BaseModel):
    """Endpoint + container metadata for a single llama.cpp model role."""

    endpoint: str
    container_name: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "ai-orchestrator"
    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 8001

    # Router API base URL. The llama.cpp router owns model loading/unloading;
    # the orchestrator only selects the logical model identifier.
    model_router_url: str = "http://llama-router:8080/v1"
    llama_cpp_api_key: str | None = None

    # Logical model-role mapping to router model identifiers.
    controller_model_name: str = "controller"
    reasoning_model_name: str = "expert"
    coder_model_name: str = "expert"
    vision_model_name: str = "vision"
    embedding_model: str = "nomic-embed-text"

    # Legacy lifecycle tuning kept only for operational rollback compatibility.
    container_start_timeout_s: float = 120.0
    health_poll_interval_s: float = 2.0
    health_timeout_s: float = 120.0

    controller_keep_alive: str = "30m"
    reasoning_keep_alive: str = "15m"
    coder_keep_alive: str = "30m"
    vision_keep_alive: str = "30m"
    controller_model_think: bool = False
    reasoning_model_think: bool = True
    controller_plan_temperature: float = 0.05
    controller_validate_temperature: float = 0.0
    controller_finalize_temperature: float = 0.15
    reasoning_temperature: float = 0.2
    controller_plan_max_tokens: int = 256
    controller_validate_max_tokens: int = 128
    controller_finalize_max_tokens: int = 1200
    reasoning_max_tokens: int = 4096
    coder_max_tokens: int = 1200
    vision_max_tokens: int = 1200

    # Knowledge service
    knowledge_service_url: str = "http://knowledge-service:8000"
    knowledge_collection: str = "homelab-knowledge"
    knowledge_top_k: int = 6
    knowledge_candidate_limit: int = 12
    knowledge_neighbor_window: int = 1
    knowledge_min_score: float = 0.55
    knowledge_min_hits: int = 1

    # Native web retrieval
    web_search_enabled: bool = True
    web_search_url: str = "http://searxng:8080"
    web_search_timeout_s: float = 20.0
    web_search_max_results: int = 10
    web_search_language: str = "en"
    web_search_safesearch: int = 0
    web_search_categories: str = "general"

    # Vision pipeline
    vision_enabled: bool = True
    vision_max_images: int = 8
    vision_timeout_s: float = 300.0
    vision_fetch_base_url: str = "http://open-webui:8080"
    vision_inject_analysis_as_system: bool = True

    # MCP / tools
    mcp_enabled: bool = True
    mcp_servers_json: str = "[]"

    # Checkpoint / storage
    checkpoint_sqlite_path: str = "/data/checkpoints.sqlite3"

    # Runtime limits
    # Conversation-history token budget (NOT the total model context budget).
    # Conversation history is trimmed oldest-first until it fits this budget.
    # Increased from 12000 to 32000 to support larger document processing
    max_context_history_tokens: int = 32000

    # Document-specific token budget for file attachments.
    # When processing large PDFs/documents, the knowledge service may generate
    # substantial context. This budget determines how much document context
    # is retained alongside conversation history.
    max_document_context_tokens: int = 16000

    # File size limits (in bytes)
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10 MB default
    max_files_per_request: int = 10
    request_timeout_s: float = 300.0

    # OpenAI-compatible surface
    openai_api_key: SecretStr | None = None
    openai_organization: str | None = None

    # Feature flags
    enable_rag: bool = True
    enable_vision: bool = True

    # Persistent conversation evidence (Feature 3)
    # Bounds for reusable specialist evidence stored in the LangGraph
    # checkpoint. These are operational limits; the state model itself stays a
    # simple container.
    conversation_evidence_max_items: int = 12
    conversation_evidence_max_content_length: int = 4000
    conversation_evidence_max_total_chars: int = 24000

    @property
    def reasoning_think(self) -> bool:
        return self.reasoning_model_think


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
