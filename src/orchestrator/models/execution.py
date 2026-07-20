from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from orchestrator.common.enums import (
    ControllerAction,
    RouteType,
    SpecialistType,
)


class ExecutionPlan(BaseModel):
    """
    Immutable execution plan produced by the planner.

    Once created, this object should never be modified.

    Architectural note:
    - `route` is a coarse orchestration path (graph route) and MUST NOT include
      specialist steps such as `REASONING`.
    - specialist work is represented only via `execution_queue`.
    """

    classification: str = "GENERAL"

    confidence: float = 0.0

    # Coarse orchestration route only.
    # Reasoning must be represented as a specialist step in `execution_queue`.
    route: RouteType = RouteType.GENERAL


    requires_repository: bool = False

    requires_web: bool = False

    requires_reasoning: bool = False

    requires_code: bool = False

    requires_tools: bool = False

    requires_vision: bool = False

    execution_queue: list[SpecialistType] = Field(default_factory=list)

    tool_requests: list[dict[str, Any]] = Field(default_factory=list)


class RetryState(BaseModel):
    """
    Tracks retries for each specialist.
    """

    attempts: int = 0

    max_attempts: int = 1

    last_error: str = ""

    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts


class RuntimeState(BaseModel):
    """
    Mutable runtime state.

    Only the execution engine modifies this.
    """

    queue: list[SpecialistType] = Field(default_factory=list)

    current_index: int = 0

    current_specialist: SpecialistType | None = None

    completed: list[SpecialistType] = Field(default_factory=list)

    retries: dict[SpecialistType, RetryState] = Field(default_factory=dict)

    started_at: datetime | None = None

    finished_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.current_index >= len(self.queue)

    @property
    def pending(self) -> list[SpecialistType]:
        return self.queue[self.current_index :]

    def start(self) -> None:
        if self.started_at is None:
            self.started_at = datetime.utcnow()

        if self.queue:
            self.current_specialist = self.queue[self.current_index]

    def advance(self) -> None:
        if self.finished:
            return

        specialist = self.queue[self.current_index]

        self.completed.append(specialist)

        self.current_index += 1

        if self.finished:
            self.current_specialist = None
            self.finished_at = datetime.utcnow()
            return

        self.current_specialist = self.queue[self.current_index]

    def retry(self, specialist: SpecialistType, error: str = "") -> bool:
        state = self.retries.setdefault(
            specialist,
            RetryState(),
        )

        if not state.can_retry():
            return False

        state.attempts += 1
        state.last_error = error

        self.current_specialist = specialist

        return True


class ValidationResult(BaseModel):
    """
    Produced by the validator after each specialist.

    Validator ONLY inspects evidence.

    It never edits the execution queue.
    """

    action: ControllerAction = ControllerAction.CONTINUE

    confidence: float = 0.0

    complete: bool = False

    summary: str = ""

    retry: bool = False

    retry_reason: str = ""

    requires_reasoning: bool = False

    requires_clarification: bool = False

    fallback_to_general: bool = False

    reason: str = ""

    issues: list[str] = Field(default_factory=list)

    notes: str = ""


class ExecutionState(BaseModel):
    """
    Complete execution state.

    Planner owns:
        plan

    Execution engine owns:
        runtime

    Validator returns:
        validation
    """

    plan: ExecutionPlan = Field(
        default_factory=lambda: ExecutionPlan(
            classification="GENERAL",
        )
    )

    runtime: RuntimeState = Field(default_factory=RuntimeState)

    validation: ValidationResult | None = None

    def initialize(self) -> None:
        """
        Initializes runtime from the immutable plan.
        """

        self.runtime.queue = list(self.plan.execution_queue)

        self.runtime.current_index = 0

        self.runtime.completed.clear()

        self.runtime.current_specialist = (
            self.runtime.queue[0]
            if self.runtime.queue
            else None
        )

        self.runtime.started_at = datetime.utcnow()

    @property
    def current_specialist(self) -> SpecialistType | None:
        return self.runtime.current_specialist

    @property
    def finished(self) -> bool:
        return self.runtime.finished
