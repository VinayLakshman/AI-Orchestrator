from __future__ import annotations

from enum import Enum, StrEnum


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RouteType(str, Enum):
    GENERAL = "general"
    VISION = "vision"
    CODE = "code"
    RAG = "rag"
    TOOLS = "tools"
    MULTI_STEP = "multi_step"
    CLARIFY = "clarify"


class KnowledgeServicePolicy(StrEnum):
    NORMAL = "normal"
    REQUIRED = "required"


class SpecialistType(StrEnum):
    KNOWLEDGE = "knowledge"
    WEB = "web"
    VISION = "vision"
    CODER = "coder"
    TOOLS = "tools"
    REASONING = "reasoning"
    CLARIFY = "clarify"
    IMAGE_GENERATION = "image_generation"


class ControllerAction(StrEnum):
    CONTINUE = "continue"
    FINALIZE = "finalize"
    REASON = "reason"
    CLARIFY = "clarify"


class VisionTaskType(StrEnum):
    OCR = "ocr"
    SCREENSHOT = "screenshot"
    TERMINAL = "terminal"
    CHART = "chart"
    DIAGRAM = "diagram"
    DOCUMENT = "document"
    PHOTO = "photo"
    MIXED = "mixed"
