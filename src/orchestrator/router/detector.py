from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import RouteDecision, RouteType


IMAGE_PATTERNS = (
    r"\bimage\b",
    r"\bscreenshot\b",
    r"\bphoto\b",
    r"\bdiagram\b",
    r"\bocr\b",
    r"\banalyze this\b",
    r"\bwhat is in this\b",
    r"\bwhat do you see\b",
)

CODE_PATTERNS = (
    r"\bcode\b",
    r"\bpython\b",
    r"\bjavascript\b",
    r"\btypescript\b",
    r"\bdockerfile\b",
    r"\bdocker compose\b",
    r"\bcompose file\b",
    r"\bshell script\b",
    r"\bdebug\b",
    r"\berror\b",
    r"\btraceback\b",
    r"\bstack trace\b",
)

RAG_PATTERNS = (
    r"\bmy knowledge\b",
    r"\bmy docs\b",
    r"\bsearch my\b",
    r"\bin my repo\b",
    r"\bin the docs\b",
    r"\bwhat does my\b",
    r"\baccording to\b",
    r"\bfind in\b",
)

TOOL_PATTERNS = (
    r"\brestart\b",
    r"\bstart\b",
    r"\bstop\b",
    r"\bdeploy\b",
    r"\bproxmox\b",
    r"\bhome assistant\b",
    r"\bdocker\b",
    r"\bgit\b",
    r"\bmcp\b",
    r"\brun command\b",
    r"\bexecute\b",
)

MULTISTEP_PATTERNS = (
    r"\band then\b",
    r"\bafter that\b",
    r"\bcompare\b",
    r"\bcombine\b",
    r"\bfirst\b.*\bthen\b",
    r"\banalyze\b.*\bthen\b",
)


@dataclass(slots=True)
class DetectionSignals:
    image: bool = False
    code: bool = False
    rag: bool = False
    tools: bool = False
    multi_step: bool = False
    explicit_tool_command: bool = False


def _match_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def detect_signals(user_text: str) -> DetectionSignals:
    text = user_text.strip()
    return DetectionSignals(
        image=_match_any(text, IMAGE_PATTERNS),
        code=_match_any(text, CODE_PATTERNS),
        rag=_match_any(text, RAG_PATTERNS),
        tools=_match_any(text, TOOL_PATTERNS),
        multi_step=_match_any(text, MULTISTEP_PATTERNS),
        explicit_tool_command=text.startswith("/"),
    )


def deterministic_route(user_text: str) -> RouteDecision | None:
    signals = detect_signals(user_text)

    if signals.explicit_tool_command:
        return RouteDecision(
            route=RouteType.TOOLS,
            confidence=0.99,
            reason="Explicit slash command detected.",
            needs_tools=True,
        )

    if signals.image:
        return RouteDecision(
            route=RouteType.VISION,
            confidence=0.95,
            reason="Image/vision intent detected from prompt text.",
            needs_vision=True,
            candidate_models=[],
        )

    if signals.multi_step:
        return RouteDecision(
            route=RouteType.MULTI_STEP,
            confidence=0.84,
            reason="Prompt suggests multiple sequential tasks.",
            needs_planning=True,
            needs_rag=signals.rag,
            needs_tools=signals.tools,
            needs_code=signals.code,
        )

    if signals.tools and not signals.code and not signals.rag:
        return RouteDecision(
            route=RouteType.TOOLS,
            confidence=0.83,
            reason="Direct tool/execution intent detected.",
            needs_tools=True,
        )

    if signals.rag and not signals.code:
        return RouteDecision(
            route=RouteType.RAG,
            confidence=0.8,
            reason="Knowledge/retrieval intent detected.",
            needs_rag=True,
        )

    if signals.code:
        return RouteDecision(
            route=RouteType.CODE,
            confidence=0.78,
            reason="Coding/debugging intent detected.",
            needs_code=True,
        )

    return None