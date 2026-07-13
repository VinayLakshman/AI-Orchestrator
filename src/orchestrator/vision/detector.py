from __future__ import annotations

from ..common.enums import VisionTaskType


def infer_vision_task(user_text: str) -> VisionTaskType:
    text = (user_text or "").lower()

    if any(token in text for token in ("ocr", "read text", "transcribe", "exact text")):
        return VisionTaskType.OCR
    if any(token in text for token in ("terminal", "log", "error", "stack trace")):
        return VisionTaskType.TERMINAL
    if any(token in text for token in ("chart", "graph", "plot", "trend")):
        return VisionTaskType.CHART
    if any(token in text for token in ("diagram", "architecture", "topology", "flow")):
        return VisionTaskType.DIAGRAM
    if any(token in text for token in ("document", "pdf", "page", "paper")):
        return VisionTaskType.DOCUMENT
    if any(token in text for token in ("screenshot", "ui", "interface", "screen")):
        return VisionTaskType.SCREENSHOT
    return VisionTaskType.MIXED
