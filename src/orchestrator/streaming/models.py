from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StreamKind(StrEnum):
    GRAPH_STARTED = "graph_started"
    GRAPH_FINISHED = "graph_finished"
    GRAPH_FAILED = "graph_failed"

    CONTROLLER_STARTED = "controller_started"
    CONTROLLER_PLAN = "controller_plan"
    CONTROLLER_VALIDATED = "controller_validated"

    KNOWLEDGE_STARTED = "knowledge_started"
    KNOWLEDGE_FINISHED = "knowledge_finished"

    VISION_STARTED = "vision_started"
    VISION_PROGRESS = "vision_progress"
    VISION_FINISHED = "vision_finished"

    CODE_STARTED = "code_started"
    CODE_FINISHED = "code_finished"

    REASONING_STARTED = "reasoning_started"
    REASONING_TOKEN = "reasoning_token"
    REASONING_FINISHED = "reasoning_finished"

    LLM_STARTED = "llm_started"
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

        return (
            f"id: {self.seq}\n"
            f"event: {self.kind.value}\n"
            f"data: {json.dumps(self.model_dump(mode='json'))}\n\n"
        )