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

    # Llama CPP default base URL (fallback when no per-model endpoint is set).
    llama_cpp_base_url: str = "http://llama-controller:8081"
    llama_controller_model: str = "controller"
    llama_cpp_api_key: str | None = None

    # Structured per-role model configuration. Each role runs its own llama.cpp
    # server container exposing an OpenAI-compatible API.
    models: dict[str, LlamaCppModelConfig] = {
        "controller": LlamaCppModelConfig(
            endpoint="http://llama-controller:8080",
            container_name="llama-controller",
        ),
        "reasoning": LlamaCppModelConfig(
            endpoint="http://llama-reasoning:8080",
            container_name="llama-reasoning",
        ),
        "coder": LlamaCppModelConfig(
            endpoint="http://llama-coder:8080",
            container_name="llama-coder",
        ),
        "vision": LlamaCppModelConfig(
            endpoint="http://llama-vision:8080",
            container_name="llama-vision",
        ),
    }

    # Container lifecycle / health tuning.
    container_start_timeout_s: float = 120.0
    health_poll_interval_s: float = 2.0
    health_timeout_s: float = 120.0

    controller_model: str = "controller"
    reasoning_model: str = "reasoning"
    coder_model: str = "coder"
    vision_model: str = "vision"
    embedding_model: str = "nomic-embed-text"

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
    max_context_messages: int = 24
    max_controller_cycles: int = 6
    max_specialist_executions: int = 12
    max_specialist_retries: int = 1
    workflow_stall_limit: int = 2
    max_tool_rounds: int = 4
    request_timeout_s: float = 300.0

    # OpenAI-compatible surface
    openai_api_key: SecretStr | None = None
    openai_organization: str | None = None

    # Feature flags
    enable_streaming: bool = True
    enable_rag: bool = True
    enable_vision: bool = True

    @property
    def general_model(self) -> str:
        return self.controller_model

    @property
    def controller_temperature(self) -> float:
        return self.controller_plan_temperature

    @property
    def controller_max_tokens(self) -> int:
        return self.controller_plan_max_tokens

    @property
    def controller_final_max_tokens(self) -> int:
        return self.controller_finalize_max_tokens

    @property
    def controller_think(self) -> bool:
        return self.controller_model_think

    @property
    def reasoning_think(self) -> bool:
        return self.reasoning_model_think

    @property
    def router_model(self) -> str | None:
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
