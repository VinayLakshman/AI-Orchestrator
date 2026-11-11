from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the focused repository RAG API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_prefix="ZION_",
    )

    app_name: str = "zion-git-rag"
    env: str = "dev"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8001

    llm_base_url: str = "http://192.168.1.61:8081/v1"
    llm_model: str = "expert"
    llm_api_key: str | None = None
    llm_timeout_s: float = 180.0
    answer_max_tokens: int = 1800
    answer_temperature: float = 0.15

    knowledge_service_url: str = "http://knowledge-service:8000"
    knowledge_service_api_key: str | None = None
    knowledge_timeout_s: float = 45.0
    knowledge_top_k: int = 6
    knowledge_candidate_limit: int = 12
    knowledge_neighbor_window: int = 1
    max_context_chars: int = 24000

    api_key: str | None = None
    require_auth: bool = False

    @property
    def model_router_url(self) -> str:
        """Compatibility alias consumed by the existing OpenAI-compatible client."""
        return self.llm_base_url

    @property
    def llama_cpp_api_key(self) -> str | None:
        return self.llm_api_key

    @property
    def request_timeout_s(self) -> float:
        return self.llm_timeout_s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
