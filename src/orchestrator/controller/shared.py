from __future__ import annotations

from typing import Any, Iterable

from ..common.enums import RouteType, SpecialistType
from ..schemas import ControllerPlan, RouteDecision


def normalize_specialist(value: Any) -> SpecialistType | None:
    if not value:
        return None
    try:
        return SpecialistType(str(value))
    except Exception:
        return None


def unique_specialist_values(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        step = normalize_specialist(value)
        if step is None:
            continue
        if step.value in seen:
            continue
        seen.add(step.value)
        normalized.append(step.value)
    return normalized


def retry_counts_from_state(state: dict[str, Any]) -> dict[str, int]:
    raw = state.get("retry_counts", {}) or {}
    counts: dict[str, int] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            step = normalize_specialist(key)
            if step is None:
                continue
            try:
                counts[step.value] = max(0, int(value))
            except Exception:
                counts[step.value] = 0
    return counts


def current_execution_plan(state: dict[str, Any]) -> ControllerPlan | None:
    value = state.get("execution_plan") or state.get("controller_validation") or state.get("controller_plan")
    if isinstance(value, ControllerPlan):
        return value
    if isinstance(value, dict):
        try:
            return ControllerPlan.model_validate(value)
        except Exception:
            return None
    return None


def current_pending_steps(state: dict[str, Any]) -> list[str]:
    plan = current_execution_plan(state)
    if plan and plan.pending_specialists:
        return unique_specialist_values(plan.pending_specialists)
    pending = list(state.get("pending_specialists", []) or [])
    if not pending:
        pending = list(state.get("pending_steps", []) or [])
    return unique_specialist_values(pending)


def step_from_pending(pending_steps: list[str] | None) -> SpecialistType | None:
    if not pending_steps:
        return None
    try:
        return SpecialistType(pending_steps[0])
    except Exception:
        return None


def current_executed_steps(state: dict[str, Any]) -> list[str]:
    return unique_specialist_values(state.get("executed_specialists", []) or [])


def current_failed_steps(state: dict[str, Any]) -> list[str]:
    return unique_specialist_values(state.get("failed_specialists", []) or [])


def has_finalize_path(state: dict[str, Any]) -> bool:
    plan = current_execution_plan(state)
    if plan is not None:
        return bool(plan.complete)
    return not current_pending_steps(state) and not bool(state.get("needs_reasoning")) and not bool(
        state.get("requires_clarification")
    )


def plan_to_route(plan: ControllerPlan) -> RouteDecision:
    pending = list(plan.pending_specialists or [])

    if plan.complete:
        route = RouteType.GENERAL

    elif len(pending) > 1:
        route = RouteType.MULTI_STEP

    elif not pending:
        route = RouteType.GENERAL

    else:
        specialist = pending[0]

        if specialist == SpecialistType.KNOWLEDGE:
            route = RouteType.RAG

        elif specialist == SpecialistType.VISION:
            route = RouteType.VISION

        elif specialist == SpecialistType.CODER:
            route = RouteType.CODE

        elif specialist == SpecialistType.TOOLS:
            route = RouteType.TOOLS

        elif specialist == SpecialistType.CLARIFY:
            route = RouteType.CLARIFY

        elif specialist in (
            SpecialistType.WEB,
            SpecialistType.REASONING,
        ):
            route = RouteType.MULTI_STEP

        else:
            route = RouteType.GENERAL

    return RouteDecision(
        route=route,
        confidence=plan.confidence,
        reason=plan.summary,
        needs_rag=SpecialistType.KNOWLEDGE in pending,
        needs_vision=SpecialistType.VISION in pending,
        needs_code=SpecialistType.CODER in pending,
        needs_tools=SpecialistType.TOOLS in pending,
        needs_planning=len(pending) > 1,
        candidate_models=[],
    )
