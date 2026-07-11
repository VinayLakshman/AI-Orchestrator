from __future__ import annotations

from typing import Any, TypedDict


class OrchestratorState(TypedDict, total=False):
    thread_id: str
    messages: list[dict[str, Any]]
    route: dict[str, Any]
    route_name: str

    vision: dict[str, Any]
    vision_context: str
    vision_task: str
    vision_confidence: float
    vision_image_hashes: list[str]
    vision_cache_hit: bool

    knowledge: list[dict[str, Any]]
    knowledge_context: str
    retrieval_stats: dict[str, Any]

    used_models: list[str]
    used_tools: list[str]
    answer: str
    metadata: dict[str, Any]
    error: str