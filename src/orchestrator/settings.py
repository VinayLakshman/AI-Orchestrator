from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Ollama / local models
    ollama_base_url: str = "http://ollama:11434"
    ollama_keep_alive: str = "30m"
    ollama_num_parallel: int = 1
    ollama_max_loaded_models: int = 2

    controller_model: str = "qwen3:4b"
    reasoning_model: str = "qwen3:14b"
    coder_model: str = "qwen2.5-coder:7b"
    vision_model: str = "qwen2.5-vl:7b"
    embedding_model: str = "nomic-embed-text"

    controller_keep_alive: str = "30m"
    reasoning_keep_alive: str = "15m"
    controller_model_think: bool = False
    reasoning_model_think: bool = True
    controller_temperature: float = 0.15
    reasoning_temperature: float = 0.2
    controller_max_tokens: int = 1024
    reasoning_max_tokens: int = 1600
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
