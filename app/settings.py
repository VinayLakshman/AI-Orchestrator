from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = Field(default='ai-orchestrator', alias='APP_NAME')
    app_env: str = Field(default='development', alias='APP_ENV')
    app_host: str = Field(default='0.0.0.0', alias='APP_HOST')
    app_port: int = Field(default=8000, alias='APP_PORT')
    log_level: str = Field(default='INFO', alias='LOG_LEVEL')

    ollama_base_url: str = Field(default='http://host.docker.internal:11434', alias='OLLAMA_BASE_URL')
    ollama_main_model: str = Field(default='qwen3:14b', alias='OLLAMA_MAIN_MODEL')
    ollama_coder_model: str = Field(default='qwen2.5-coder:7b', alias='OLLAMA_CODER_MODEL')
    ollama_vision_model: str = Field(default='qwen2.5-vl:7b', alias='OLLAMA_VISION_MODEL')

    knowledge_service_url: str = Field(default='http://host.docker.internal:8001', alias='KNOWLEDGE_SERVICE_URL')
    knowledge_retrieve_path: str = Field(default='/retrieve', alias='KNOWLEDGE_RETRIEVE_PATH')

    mcp_gateway_url: str = Field(default='', alias='MCP_GATEWAY_URL')
    mcp_execute_path: str = Field(default='/tools/execute', alias='MCP_EXECUTE_PATH')

    request_timeout_seconds: float = Field(default=60.0, alias='REQUEST_TIMEOUT_SECONDS')
    max_context_chunks: int = Field(default=8, alias='MAX_CONTEXT_CHUNKS')
    enable_streaming: bool = Field(default=True, alias='ENABLE_STREAMING')
    cors_allow_origins: str = Field(default='*', alias='CORS_ALLOW_ORIGINS')

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
