from __future__ import annotations

from typing import Any

from ..common.enums import ControllerAction, KnowledgeServicePolicy
from ..models.evidence import EvidenceLedger
from ..models.state import OrchestratorState
from . import get_logger


logger = get_logger(__name__)

_ROW_WIDTH = 15
_PROMPT_LIMIT = 100
_TRACE_EXCLUDE = {"prepare"}


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
    return f"({duration} ms)"


def _truncate_prompt(text: str | None) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if len(cleaned) <= _PROMPT_LIMIT:
        return cleaned
    return cleaned[: max(0, _PROMPT_LIMIT - 3)].rstrip() + "..."


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


def _knowledge_service_policy(state: OrchestratorState) -> str:
    policy = getattr(
        state.request,
        "knowledge_service_policy",
        KnowledgeServicePolicy.NORMAL,
    )
    if isinstance(policy, KnowledgeServicePolicy):
        return policy.value
    return (
        str(policy or KnowledgeServicePolicy.NORMAL.value).strip()
        or KnowledgeServicePolicy.NORMAL.value
    )


def _image_urls_from_metadata(state: OrchestratorState) -> list[str]:
    """Valid generated image URLs from the terminal image-generation route."""
    if state.response.metadata.get("route") != "image_generation":
        return []
    raw = state.response.metadata.get("image_urls")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(url).strip() for url in raw if isinstance(url, str) and str(url).strip()]


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

    # The image-generation route is intentionally terminal (no finalizer), so
    # an empty ``final_response`` is EXPECTED there. Success is judged by the
    # authoritative image result, not by arbitrary metadata presence.
    generated_images = _image_urls_from_metadata(state)
    if generated_images:
        return "SUCCESS"

    if not str(state.response.final_response or "").strip():
        return "FAILED"

    return "SUCCESS"


def _execution_trace(
    state: OrchestratorState,
    execution_trace: list[dict[str, Any]] | None,
    timings: dict[str, float] | None,
) -> list[dict[str, Any]]:
    trace = list(execution_trace or state.debug.execution_trace or [])
    if trace:
        return trace

    fallback: list[dict[str, Any]] = []
    for key, value in (timings or {}).items():
        label = str(key).replace("_", " ").strip().title()
        if key in _TRACE_EXCLUDE:
            continue
        fallback.append({"key": key, "label": label, "duration_ms": value})
    return fallback


def _execution_path_lines(
    state: OrchestratorState,
    execution_trace: list[dict[str, Any]] | None,
    timings: dict[str, float] | None,
) -> list[str]:
    lines: list[str] = []
    for entry in _execution_trace(state, execution_trace, timings):
        label = str(entry.get("label") or entry.get("key") or "").strip()
        if not label or label.lower() in _TRACE_EXCLUDE:
            continue
        duration = _duration_value(entry.get("duration_ms"))
        prefix = "→ " if lines else ""
        lines.append(f"{prefix}{label} {duration}")
    return lines


def _evidence_stats(state: OrchestratorState) -> dict[str, int]:
    evidence: EvidenceLedger = state.evidence
    image_count = 0
    if evidence.vision is not None:
        metadata = evidence.vision.metadata or {}
        try:
            image_count = int(metadata.get("image_count") or 0)
        except (TypeError, ValueError):
            image_count = 0

    # Generated (ComfyUI/Open WebUI) images take precedence and are counted
    # from the authoritative response metadata; the comfyui evidence ledger is
    # a secondary source used only when the metadata is absent. Input images
    # are only counted when nothing was generated, so no image is counted twice.
    generated_images = _image_urls_from_metadata(state)
    if generated_images:
        image_count = len(generated_images)
    else:
        generated_count = evidence.comfyui_image_count
        if generated_count:
            image_count = generated_count
        elif not image_count:
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
    execution_trace: list[dict[str, Any]] | None,
    timings: dict[str, float] | None,
    total_duration_ms: int | float | None,
) -> str:
    evidence = _evidence_stats(state)
    prompt = _truncate_prompt(state.request.user_message)
    path_lines = _execution_path_lines(state, execution_trace, timings)

    lines = [
        "Request Summary",
        "---------------",
        _format_row("Request ID", request_id or str(state.request.request_id or "")),
        _format_row("Prompt", prompt),
        _format_row("Classification", str(state.execution.plan.classification or "GENERAL")),
        _format_row("Knowledge Policy", _knowledge_service_policy(state)),
        "",
        "Execution Path",
        "--------------",
    ]

    lines.extend(path_lines or ["(no executed nodes recorded)"])

    lines.extend(
        [
            "",
            _format_row("Total", f"{int(round(float(total_duration_ms or 0)))} ms"),
            "",
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
    execution_trace: list[dict[str, Any]] | None,
    timings: dict[str, float] | None,
    total_duration_ms: int | float | None,
) -> None:
    logger.info(
        "%s",
        build_request_summary(
            request_id=request_id,
            state=state,
            execution_trace=execution_trace,
            timings=timings,
            total_duration_ms=total_duration_ms,
        ),
    )
