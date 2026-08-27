from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel
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

    # Max time to wait for the controller model to become READY in
    # llama-router after the GPU has been released by ComfyUI.  This is the
    # post-image-generation handoff: the controller must be verifiably loaded
    # before controller.validate() runs.
    controller_load_timeout_s: float = 300.0

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

    # Knowledge service
    knowledge_service_url: str = "http://knowledge-service:8000"
    knowledge_top_k: int = 6
    knowledge_candidate_limit: int = 12
    knowledge_neighbor_window: int = 1

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

    # Checkpoint / storage
    checkpoint_sqlite_path: str = "/data/checkpoints.sqlite3"

    # Runtime limits
    # Conversation-history token budget (NOT the total model context budget).
    # Conversation history is trimmed oldest-first until it fits this budget.
    max_context_history_tokens: int = 12000
    request_timeout_s: float = 300.0

    # Feature flags
    enable_rag: bool = True
    enable_vision: bool = True

    # Image generation (delegated to Open WebUI)
    # Open WebUI is the single source of truth for all image-generation
    # configuration (engine, ComfyUI workflow, node mappings, model, size,
    # steps). The orchestrator only needs the API boundary and a credential.
    openwebui_base_url: str = "http://open-webui:8080"
    # Legacy setting kept for configuration compatibility. Generated image
    # URLs are now delivered as RELATIVE Open WebUI paths (the browser resolves
    # them against the public origin), so this value is no longer used for
    # browser-facing image URL normalization. It is retained only as a
    # recognized service origin that may be stripped from absolute generated
    # file URLs.
    openwebui_public_base_url: str | None = None
    openwebui_api_key: str | None = None
    image_generation_timeout: float = 300.0  # 5 minutes

    # Max time any requester waits for the GPU while image generation owns or
    # has reserved it (sentinel owners "comfyui" / "transitioning_to_comfyui").
    gpu_ownership_wait_timeout_s: float = 1800.0

    # ComfyUI GPU-release barrier: after an Open WebUI generation completes,
    # llama-router must NOT load a model until ComfyUI's VRAM is verifiably
    # released. The barrier ACTIVELY RETRIES POST /free: invoke /free, verify
    # real memory observables against the captured baseline, wait
    # comfyui_free_retry_interval_s if not yet released, and repeat until
    # verified or the bounded timeout expires.
    comfyui_release_timeout_s: float = 120.0
    # Interval between successive /free attempts when verification has NOT yet
    # succeeded (NOT a telemetry-poll period). The first /free fires
    # immediately; verification happens right after each attempt.
    comfyui_free_retry_interval_s: float = 5.0
    # Tolerance below the pre-generation free-VRAM baseline (MB). The baseline
    # is captured AFTER the LLM unload at acquisition time, so this compares a
    # host against itself — there are no hard-coded absolute VRAM numbers.
    comfyui_release_memory_tolerance_mb: float = 512.0

    # ComfyUI lifecycle endpoint (REQUIRED for the release barrier). This is
    # used ONLY by the GPU-release barrier: POST /free (unload_models +
    # free_memory) before verification and GET /system_stats as a direct
    # ComfyUI/VRAM memory observable. It must NEVER be used to build or
    # submit image-generation workflows directly.
    comfyui_lifecycle_base_url: str = "http://comfyui:8188"
    comfyui_free_timeout_s: float = 30.0

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
