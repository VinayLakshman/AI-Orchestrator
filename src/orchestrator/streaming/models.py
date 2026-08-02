from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StreamKind(StrEnum):
    GRAPH_STARTED = "graph_started"
    GRAPH_FINISHED = "graph_finished"
    GRAPH_FAILED = "graph_failed"

    SPECIALIST_STARTED = "specialist_started"
    SPECIALIST_PROGRESS = "specialist_progress"
    SPECIALIST_FINISHED = "specialist_finished"

    LLM_STARTED = "llm_started"
    LLM_TOKEN = "llm_token"
    LLM_FINISHED = "llm_finished"

    ERROR = "error"


class StreamEvent(BaseModel):
    seq: int
    kind: StreamKind
    request_id: str
    conversation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        import json
        from ..serialization import sanitize_for_json, validate_json_serializable

        payload = self.model_dump(mode="json")
        safe = sanitize_for_json(payload)
        validate_json_serializable(safe)
        return (
            f"id: {self.seq}\n"
            f"event: {self.kind.value}\n"
            f"data: {json.dumps(safe, ensure_ascii=False)}\n\n"
        )