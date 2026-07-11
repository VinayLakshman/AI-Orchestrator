from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #
    # Application
    #
    app_name: str = Field(default="ai-orchestrator", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    #
    # Logging
    #
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
   
    #
    # Ollama
    #
    ollama_url: str = Field(
        default="http://ollama:11434",
        alias="OLLAMA_URL",
    )

    llm_main_model: str = Field(
        default="qwen3:14b",
        alias="LLM_MAIN_MODEL",
    )

    llm_coder_model: str = Field(
        default="qwen2.5-coder:7b",
        alias="LLM_CODER_MODEL",
    )

    llm_vision_model: str = Field(
        default="qwen2.5-vl:7b",
        alias="LLM_VISION_MODEL",
    )

    #
    # Internal Services
    #
    knowledge_service_url: str = Field(
        default="http://knowledge-service:8001",
        alias="KNOWLEDGE_SERVICE_URL",
    )

    mcp_url: str = Field(
        default="http://mcp-gateway:9000",
        alias="MCP_URL",
    )

    #
    # Runtime
    #
    request_timeout_seconds: float = Field(
        default=60.0,
        alias="REQUEST_TIMEOUT_SECONDS",
    )

    max_context_chunks: int = Field(
        default=8,
        alias="MAX_CONTEXT_CHUNKS",
    )

    enable_streaming: bool = Field(
        default=True,
        alias="ENABLE_STREAMING",
    )

    #
    # API
    #
    cors_allow_origins: str = Field(
        default="*",
        alias="CORS_ALLOW_ORIGINS",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()