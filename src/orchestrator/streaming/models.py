from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class StreamKind(StrEnum):
    GRAPH_STARTED = "graph.started"
    GRAPH_FINISHED = "graph.finished"
    GRAPH_FAILED = "graph.failed"

    ROUTING_STARTED = "routing.started"
    ROUTING_FINISHED = "routing.finished"

    KNOWLEDGE_STARTED = "knowledge.started"
    KNOWLEDGE_PROGRESS = "knowledge.progress"
    KNOWLEDGE_FINISHED = "knowledge.finished"

    VISION_STARTED = "vision.started"
    VISION_PROGRESS = "vision.progress"
    VISION_FINISHED = "vision.finished"

    TOOL_STARTED = "tool.started"
    TOOL_PROGRESS = "tool.progress"
    TOOL_FINISHED = "tool.finished"

    CODE_STARTED = "code.started"
    CODE_PROGRESS = "code.progress"
    CODE_FINISHED = "code.finished"

    LLM_STARTED = "llm.started"
    LLM_TOKEN = "llm.token"
    LLM_FINISHED = "llm.finished"

    ERROR = "error"


class StreamEvent(BaseModel):
    seq: int = 0
    id: str = Field(default_factory=lambda: uuid4().hex)

    request_id: str
    conversation_id: str | None = None

    kind: StreamKind
    stage: str | None = None
    message: str

    status: Literal["info", "progress", "success", "warning", "error"] = "info"
    data: dict[str, Any] = Field(default_factory=dict)

    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_sse(self) -> str:
        payload = self.model_dump(mode="json")
        return (
            f"id: {self.seq}\n"
            f"event: {self.kind}\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        )