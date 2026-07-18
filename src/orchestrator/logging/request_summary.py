from __future__ import annotations

from typing import Any

from ..common.enums import ControllerAction
from ..models.evidence import EvidenceLedger
from ..models.state import OrchestratorState
from . import get_logger


logger = get_logger(__name__)

_ROW_WIDTH = 15


def _format_row(label: str, value: str) -> str:
    return f"{label:<{_ROW_WIDTH}}: {value}"


def _duration_value(value: Any) -> str:
    if value is None:
        return "skipped"
    try:
        duration = int(round(float(value)))
    except (TypeError, ValueError):
        return "skipped"
    if duration < 0:
        return "skipped"
    return f"{duration} ms"


def _timing_label_map() -> list[tuple[str, str]]:
    return [
        ("Prepare", "prepare"),
        ("Planner", "planner"),
        ("Knowledge", "knowledge"),
        ("Web", "web"),
        ("Vision", "vision"),
        ("Code", "coder"),
        ("Tools", "tools"),
        ("Reasoning", "reasoning"),
        ("Validation", "validation"),
        ("Clarification", "clarify"),
        ("Finalizer", "finalize"),
    ]


def _controller_model(state: OrchestratorState) -> str:
    model = str(state.execution.runtime.metadata.get("controller_model") or "").strip()
    if model:
        return model

    for used_model in state.debug.used_models:
        if used_model:
            return used_model

    return "skipped"


def _final_model(state: OrchestratorState) -> str:
    model = str(state.response.metadata.get("final_model") or "").strip()
    return model or "skipped"


def _request_status(state: OrchestratorState) -> str:
    if (
        state.response.finish_reason == "clarify"
        or state.response.metadata.get("route") == "clarify"
        or state.execution.validation is not None
        and state.execution.validation.action == ControllerAction.CLARIFY
    ):
        return "CLARIFICATION"

    if state.response.metadata.get("status") == "failed":
        return "FAILED"

    if not str(state.response.final_response or "").strip():
        return "FAILED"

    return "SUCCESS"


def _request_route(state: OrchestratorState) -> str:
    if (
        state.response.finish_reason == "clarify"
        or state.response.metadata.get("route") == "clarify"
        or state.execution.validation is not None
        and state.execution.validation.action == ControllerAction.CLARIFY
    ):
        return "clarify"

    return "finalize"


def _evidence_stats(state: OrchestratorState) -> dict[str, int]:
    evidence: EvidenceLedger = state.evidence
    image_count = 0
    if evidence.vision is not None:
        metadata = evidence.vision.metadata or {}
        try:
            image_count = int(metadata.get("image_count") or 0)
        except (TypeError, ValueError):
            image_count = 0
    if not image_count:
        image_count = len(state.request.images or [])

    return {
        "repository_hits": len(evidence.repository.primary_hits) if evidence.repository else 0,
        "web_results": len(evidence.web.results) if evidence.web else 0,
        "images": image_count,
        "tools_executed": len(evidence.tools.executions) if evidence.tools else 0,
    }


def build_request_summary(
    request_id: str,
    state: OrchestratorState,
    timings: dict[str, float] | None,
    total_duration_ms: int | float | None,
) -> str:
    timings = timings or {}
    evidence = _evidence_stats(state)

    lines = [
        "Request Summary",
        "---------------",
        _format_row("Request ID", request_id or str(state.request.request_id or "")),
        _format_row("Classification", str(state.execution.plan.classification or "GENERAL")),
        _format_row("Route", _request_route(state)),
        "",
    ]

    for label, key in _timing_label_map():
        lines.append(_format_row(label, _duration_value(timings.get(key))))

    lines.extend(
        [
            "",
            _format_row("Total", _duration_value(total_duration_ms)),
            _format_row("Controller", _controller_model(state)),
            _format_row("Final Model", _final_model(state)),
            "",
            "Evidence",
            "---------",
            _format_row("Repository Hits", str(evidence["repository_hits"])),
            _format_row("Web Results", str(evidence["web_results"])),
            _format_row("Images", str(evidence["images"])),
            _format_row("Tools Executed", str(evidence["tools_executed"])),
            "",
            _format_row("Status", _request_status(state)),
        ]
    )

    return "\n".join(lines)


def log_request_summary(
    request_id: str,
    state: OrchestratorState,
    timings: dict[str, float] | None,
    total_duration_ms: int | float | None,
) -> None:
    logger.info("%s", build_request_summary(request_id, state, timings, total_duration_ms))
