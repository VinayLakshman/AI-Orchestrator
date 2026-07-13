from __future__ import annotations

from dataclasses import dataclass

from ..schemas import RouteDecision, RouteType


@dataclass(slots=True)
class DetectedSignals:
    has_image: bool = False
    has_code: bool = False
    has_tool_request: bool = False
    has_rag_signal: bool = False
    needs_clarification: bool = False


def detect_signals(text: str) -> DetectedSignals:
    lowered = (text or "").lower()
    return DetectedSignals(
        has_image=any(token in lowered for token in ("image", "screenshot", "photo", "diagram")),
        has_code=any(token in lowered for token in ("code", "python", "javascript", "yaml", "json", "bug")),
        has_tool_request=any(token in lowered for token in ("run", "execute", "call tool", "mcp", "search the web")),
        has_rag_signal=any(token in lowered for token in ("docs", "repository", "knowledge", "retrieval", "rag")),
        needs_clarification=any(token in lowered for token in ("clarify", "which one", "what do you mean")),
    )


def deterministic_route(text: str) -> RouteDecision:
    signals = detect_signals(text)
    if signals.needs_clarification:
        route = RouteType.CLARIFY
    elif signals.has_image:
        route = RouteType.VISION
    elif signals.has_code:
        route = RouteType.CODE
    elif signals.has_tool_request:
        route = RouteType.TOOLS
    elif signals.has_rag_signal:
        route = RouteType.RAG
    else:
        route = RouteType.GENERAL

    return RouteDecision(
        route=route,
        confidence=0.5,
        reason="deterministic signal routing",
        needs_vision=signals.has_image,
        needs_rag=signals.has_rag_signal,
        needs_tools=signals.has_tool_request,
        needs_code=signals.has_code,
        needs_planning=False,
    )