from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ..common.enums import VisionTaskType


class VisionAnalysis(BaseModel):
    task_type: VisionTaskType = VisionTaskType.MIXED
    confidence: float = 0.0
    summary: str = ""
    ocr: str = ""
    layout: str = ""
    metrics: str = ""
    errors_warnings: str = ""
    observations: str = ""
    answer_context: str = ""
    image_count: int = 0
    source_model: str = ""
    raw_text: str = ""
    hashes: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class ResolvedImage:
    base64_data: str
    mime_type: str
    sha256: str
    source: str


class VisionResult(BaseModel):
    analysis: VisionAnalysis
    context_markdown: str
    cleaned_messages: list[dict[str, Any]] = Field(default_factory=list)
    image_hashes: list[str] = Field(default_factory=list)
    cache_hit: bool = False
