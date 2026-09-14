from __future__ import annotations

from datetime import datetime
from typing import Any

from orchestrator.common.enums import KnowledgeServicePolicy, SpecialistType
from orchestrator.models.chat import ChatMessage
from pydantic import BaseModel, Field

from orchestrator.models.evidence import (
    ConversationEvidenceState,
    EvidenceLedger,
)
from orchestrator.models.execution import ExecutionState


class RequestState(BaseModel):
    """
    Normalized request information.

    Populated once by the request normalizer/router.
    Never modified afterwards.
    """

    request_id: str = ""

    conversation_id: str = ""

    thread_id: str = ""

    model: str = ""

    stream: bool = False

    messages: list[ChatMessage] = Field(default_factory=list)

    user_message: str = ""

    original_query: str = ""

    resolved_query: str = ""

    is_followup: bool = False

    followup_confidence: float = 0.0

    knowledge_service_policy: KnowledgeServicePolicy = (
        KnowledgeServicePolicy.NORMAL
    )

    images: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    received_at: datetime = Field(default_factory=datetime.utcnow)


class ResponseState(BaseModel):
    """
    Output generated during orchestration.

    Only the finalizer writes final_response.
    """

    final_response: str = ""

    finish_reason: str = "stop"

    usage: dict[str, Any] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationResource(BaseModel):
    """
    Lightweight reference to a conversation-level resource.

    Identifies a resource (image, pdf, document, file) associated with the
    conversation WITHOUT storing its contents, extracted text, OCR, embeddings
    or web result payloads.

    ``resource_id`` provides a stable identity for de-duplication.
    """

    resource_id: str = ""

    resource_type: str = ""

    reference: str = ""

    name: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    """Sanitized, bounded record of one completed conversation turn."""

    turn_id: str = ""
    request_id: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: list[ChatMessage] = Field(default_factory=list)
    user_text: str = ""
    resolved_query: str = ""
    assistant_text: str = ""
    assistant_summary: str = ""
    finish_reason: str = "stop"
    intent: str = ""
    confidence: float = 0.0
    resource_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMemoryState(BaseModel):
    """Bounded checkpoint-safe memory used to answer follow-up turns."""

    schema_version: int = 1
    turns: list[ConversationTurn] = Field(default_factory=list)
    system_messages: list[ChatMessage] = Field(default_factory=list)
    developer_messages: list[ChatMessage] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    pending_clarification: dict[str, Any] = Field(default_factory=dict)
    last_answer_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    """
    Conversation-level state.

    Deliberately small. Provides the controller with a compact structured view
    of the ongoing conversation so it can make better routing decisions.

    Responsibility boundary:

    - request: RequestState  (what is happening right now)
    - conversation: ConversationState  (what the conversation is about)
    - evidence: EvidenceLedger  (what specialists discovered this request)

    ConversationState survives across requests sharing the same ``thread_id``
    via the existing LangGraph checkpoint mechanism.

    Conversation memory is bounded and sanitized. It contains only recent
    messages, compact summaries, and references; it never stores embeddings,
    raw attachments, web payloads, extracted document text, or debug prompts.
    """

    current_topic: str = ""

    topic_confidence: float = 0.0

    last_specialist: SpecialistType | None = None

    active_resources: list[ConversationResource] = Field(default_factory=list)

    has_web_results: bool = False

    last_web_query: str = ""

    last_web_at: datetime | None = None

    memory: ConversationMemoryState = Field(default_factory=ConversationMemoryState)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def resource_count(self) -> int:
        return len(self.active_resources)


class DebugState(BaseModel):
    """
    Internal debugging information.

    Never shown to the user.

    Safe to log.

    Useful for tracing controller decisions.
    """

    planner_prompt: str = ""

    planner_response: dict[str, Any] = Field(default_factory=dict)

    validator_prompt: str = ""

    validator_response: dict[str, Any] = Field(default_factory=dict)

    used_models: list[str] = Field(default_factory=list)

    used_tools: list[str] = Field(default_factory=list)

    timings: dict[str, float] = Field(default_factory=dict)

    execution_trace: list[dict[str, Any]] = Field(default_factory=list)


class OrchestratorState(BaseModel):
    """
    Canonical orchestration state.

    This object is passed through every LangGraph node.

    Nodes should ONLY modify the section they own.

request
        Immutable.

    conversation
        Conversation-level state (survives across requests sharing a thread).

    execution
        Planner + execution engine.

    evidence
        Current execution evidence (Specialists).

    conversation_evidence
        Reusable evidence across turns, persisted via the LangGraph
        checkpoint. Never reset by prepare.

    response
        Finalizer.

    debug
        Internal logging/tracing.
    """

    request: RequestState = Field(default_factory=RequestState)

    conversation: ConversationState = Field(default_factory=ConversationState)

    execution: ExecutionState = Field(default_factory=ExecutionState)

    evidence: EvidenceLedger = Field(default_factory=EvidenceLedger)

    conversation_evidence: ConversationEvidenceState = Field(
        default_factory=ConversationEvidenceState
    )

    response: ResponseState = Field(default_factory=ResponseState)

    debug: DebugState = Field(default_factory=DebugState)

    def reset_response(self) -> None:
        """
        Clears response state before a new execution.
        """

        self.response = ResponseState()

    def reset_evidence(self) -> None:
        """
        Clears accumulated evidence.
        """

        self.evidence = EvidenceLedger()

    def initialize_execution(self) -> None:
        """
        Initializes runtime execution from the immutable plan.
        """

        self.execution.initialize()

    @property
    def current_specialist(self):
        return self.execution.current_specialist

    @property
    def finished(self) -> bool:
        return self.execution.finished
