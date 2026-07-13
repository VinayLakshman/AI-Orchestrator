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


class ToolType(str, Enum):
    MCP = "mcp"
    KNOWLEDGE = "knowledge"
    OLLAMA = "ollama"


class SpecialistType(StrEnum):
    KNOWLEDGE = "knowledge"
    VISION = "vision"
    CODER = "coder"
    TOOLS = "tools"
    REASONING = "reasoning"
    CLARIFY = "clarify"


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
