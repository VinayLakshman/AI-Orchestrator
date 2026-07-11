from __future__ import annotations

import re

from .models import VisionTaskType

_OCR_PATTERNS = (
    r"\bocr\b",
    r"\bread (this|the)?\b",
    r"\btranscribe\b",
    r"\bextract text\b",
    r"\bwhat does (it|this) say\b",
    r"\bread the text\b",
)

_TERMINAL_PATTERNS = (
    r"\bterminal\b",
    r"\bconsole\b",
    r"\blog(s)?\b",
    r"\berror(s)?\b",
    r"\btraceback\b",
    r"\bstack trace\b",
    r"\bdocker\b",
    r"\bcompose\b",
    r"\bproxmox\b",
    r"\bhome assistant\b",
    r"\byaml\b",
    r"\bjson\b",
    r"\bcommand output\b",
    r"\bshell\b",
    r"\bdebug\b",
)

_SCREENSHOT_PATTERNS = (
    r"\bscreenshot\b",
    r"\bui\b",
    r"\bdashboard\b",
    r"\bwindow\b",
    r"\bpanel\b",
    r"\binterface\b",
    r"\blayout\b",
)

_CHART_PATTERNS = (
    r"\bchart\b",
    r"\bgraph\b",
    r"\bmetric(s)?\b",
    r"\btimeseries\b",
    r"\bcpu\b",
    r"\bmemory\b",
    r"\butilization\b",
    r"\btemperature\b",
    r"\btrend\b",
    r"\bgrafana\b",
)

_DIAGRAM_PATTERNS = (
    r"\bdiagram\b",
    r"\barchitecture\b",
    r"\btopology\b",
    r"\bflow\b",
    r"\bnetwork\b",
    r"\bdependency\b",
    r"\bblock diagram\b",
    r"\bsystem design\b",
)

_DOCUMENT_PATTERNS = (
    r"\bdocument\b",
    r"\bpdf\b",
    r"\bpage\b",
    r"\bscan(ned)?\b",
    r"\bpaper\b",
    r"\breport\b",
    r"\bnotes?\b",
)

_PHOTO_PATTERNS = (
    r"\bphoto\b",
    r"\bpicture\b",
    r"\bscene\b",
    r"\bperson\b",
    r"\bobject\b",
    r"\bproduct\b",
)


def _match_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def infer_vision_task(user_text: str) -> VisionTaskType:
    text = (user_text or "").strip()
    if not text:
        return VisionTaskType.MIXED

    if _match_any(text, _OCR_PATTERNS):
        return VisionTaskType.OCR

    if _match_any(text, _CHART_PATTERNS):
        return VisionTaskType.CHART

    if _match_any(text, _DIAGRAM_PATTERNS):
        return VisionTaskType.DIAGRAM

    if _match_any(text, _DOCUMENT_PATTERNS):
        return VisionTaskType.DOCUMENT

    if _match_any(text, _PHOTO_PATTERNS):
        return VisionTaskType.PHOTO

    if _match_any(text, _TERMINAL_PATTERNS):
        return VisionTaskType.TERMINAL

    if _match_any(text, _SCREENSHOT_PATTERNS):
        return VisionTaskType.SCREENSHOT

    return VisionTaskType.MIXED