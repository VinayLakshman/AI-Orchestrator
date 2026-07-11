from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VisionTaskType(str, Enum):
    OCR = "ocr"
    SCREENSHOT = "screenshot"
    TERMINAL = "terminal"
    CHART = "chart"
    DIAGRAM = "diagram"
    DOCUMENT = "document"
    PHOTO = "photo"
    MIXED = "mixed"


class ResolvedImage(BaseModel):
    ref: str
    mime_type: str = "image/png"
    sha256: str
    base64_data: str


class VisionAnalysis(BaseModel):
    task_type: VisionTaskType
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str

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


class VisionResult(BaseModel):
    analysis: VisionAnalysis
    context_markdown: str
    cleaned_messages: list[dict[str, Any]] = Field(default_factory=list)
    image_hashes: list[str] = Field(default_factory=list)
    cache_hit: bool = False