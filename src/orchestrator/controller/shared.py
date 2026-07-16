from __future__ import annotations

from typing import Any, Iterable

from ..common.enums import RouteType, SpecialistType
from ..schemas import ControllerPlan, RouteDecision


def normalize_specialist(value: Any) -> SpecialistType | None:
    if value is None:
        return None

    if isinstance(value, SpecialistType):
        return value

    try:
        return SpecialistType(str(value).strip().lower())
    except Exception:
        return None


def unique_specialists(values: Iterable[Any]) -> list[SpecialistType]:
    seen: set[SpecialistType] = set()
    result: list[SpecialistType] = []

    for value in values:
        specialist = normalize_specialist(value)
        if specialist is None:
            continue

        if specialist in seen:
            continue

        seen.add(specialist)
        result.append(specialist)

    return result


def unique_specialist_values(values: Iterable[Any]) -> list[str]:
    return [step.value for step in unique_specialists(values)]


def retry_counts_from_state(state: dict[str, Any]) -> dict[str, int]:
    raw = state.get("retry_counts") or {}

    counts: dict[str, int] = {}

    if not isinstance(raw, dict):
        return counts

    for key, value in raw.items():
        specialist = normalize_specialist(key)
        if specialist is None:
            continue

        try:
            counts[specialist.value] = max(0, int(value))
        except Exception:
            counts[specialist.value] = 0

    return counts


def current_execution_plan(state: dict[str, Any]) -> ControllerPlan | None:
    value = (
        state.get("execution_plan")
        or state.get("controller_validation")
        or state.get("controller_plan")
    )

    if value is None:
        return None

    if isinstance(value, ControllerPlan):
        return value

    if isinstance(value, dict):
        try:
            return ControllerPlan.model_validate(value)
        except Exception:
            return None

    return None


def execution_queue(plan: ControllerPlan | None) -> list[SpecialistType]:
    """
    Planner output is the source of truth.

    Fallback to pending_specialists while older plans still exist.
    """

    if plan is None:
        return []

    if getattr(plan, "execution_queue", None):
        return unique_specialists(plan.execution_queue)

    if plan.pending_specialists:
        return unique_specialists(plan.pending_specialists)

    if plan.next_specialist:
        return [plan.next_specialist]

    return []


def current_pending_steps(state: dict[str, Any]) -> list[str]:
    plan = current_execution_plan(state)

    if plan is not None:
        return [step.value for step in execution_queue(plan)]

    pending = (
        state.get("pending_specialists")
        or state.get("pending_steps")
        or []
    )

    return unique_specialist_values(pending)


def step_from_pending(
    pending_steps: list[str] | None,
) -> SpecialistType | None:
    if not pending_steps:
        return None

    return normalize_specialist(pending_steps[0])


def current_executed_steps(state: dict[str, Any]) -> list[str]:
    return unique_specialist_values(
        state.get("executed_specialists") or []
    )


def current_failed_steps(state: dict[str, Any]) -> list[str]:
    return unique_specialist_values(
        state.get("failed_specialists") or []
    )


def has_finalize_path(state: dict[str, Any]) -> bool:
    plan = current_execution_plan(state)

    if plan is not None:
        return (
            plan.complete
            or len(execution_queue(plan)) == 0
        )

    return (
        not current_pending_steps(state)
        and not bool(state.get("needs_reasoning"))
        and not bool(state.get("requires_clarification"))
    )


def plan_to_route(plan: ControllerPlan) -> RouteDecision:
    queue = execution_queue(plan)

    if plan.complete:
        route = RouteType.GENERAL

    elif not queue:
        route = RouteType.GENERAL

    elif len(queue) > 1:
        route = RouteType.MULTI_STEP

    else:
        first = queue[0]

        if first == SpecialistType.KNOWLEDGE:
            route = RouteType.RAG

        elif first == SpecialistType.VISION:
            route = RouteType.VISION

        elif first == SpecialistType.CODER:
            route = RouteType.CODE

        elif first == SpecialistType.TOOLS:
            route = RouteType.TOOLS

        elif first == SpecialistType.CLARIFY:
            route = RouteType.CLARIFY

        elif first in (
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
        needs_rag=SpecialistType.KNOWLEDGE in queue,
        needs_vision=SpecialistType.VISION in queue,
        needs_code=SpecialistType.CODER in queue,
        needs_tools=SpecialistType.TOOLS in queue,
        needs_planning=len(queue) > 1,
        candidate_models=[],
    )