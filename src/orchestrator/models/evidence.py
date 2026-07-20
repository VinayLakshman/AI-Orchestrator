from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RepositoryEvidence(BaseModel):
    repository: str | None = None
    branch: str | None = None
    commit: str | None = None

    question: str | None = None
    retrieval_reason: str |None = None

    confidence: float = 0.0

    context: str = ""

    hit_count: int = 0

    primary_hits: list[dict[str, Any]] = Field(default_factory=list)

    expanded_hits: list[dict[str, Any]] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class WebEvidence(BaseModel):
    query: str | None = None

    confidence: float = 0.0

    summary: str = ""

    results: list[dict[str, Any]] = Field(default_factory=list)

    snippets: list[str] = Field(default_factory=list)

    urls: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionEvidence(BaseModel):
    task: str | None = None

    confidence: float = 0.0

    summary: str = ""

    context: str = ""

    observations: list[str] = Field(default_factory=list)

    extracted_text: str = ""

    detected_objects: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeEvidence(BaseModel):
    language: str | None = None

    task: str = ""

    summary: str = ""

    generated_code: str = ""

    explanation: str = ""

    files: list[str] = Field(default_factory=list)

    tests: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)

    confidence: float = 0.0

    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecution(BaseModel):
    tool_name: str

    success: bool

    summary: str = ""

    inputs: dict[str, Any] = Field(default_factory=dict)

    outputs: dict[str, Any] = Field(default_factory=dict)

    duration_ms: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolEvidence(BaseModel):
    """
    Evidence produced by the Tools specialist.
    """

    executions: list[ToolExecution] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class ReasoningEvidence(BaseModel):
    """
    Evidence produced by the Reasoning specialist.

    This is NOT chain-of-thought.

    It stores only synthesized conclusions that may be safely
    consumed by the finalizer.
    """

    summary: str = ""

    conclusions: list[str] = Field(default_factory=list)

    assumptions: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceLedger(BaseModel):
    """
    Canonical evidence store.

    Every specialist owns exactly one section.

    Planner:
        writes nothing

    Validator:
        reads everything
        writes nothing

    Finalizer:
        reads everything
        writes nothing
    """

    repository: RepositoryEvidence | None = None

    web: WebEvidence | None = None

    vision: VisionEvidence | None = None

    code: CodeEvidence | None = None

    tools: ToolEvidence | None = None

    reasoning: ReasoningEvidence | None = None

    @property
    def has_repository(self) -> bool:
        return self.repository is not None

    @property
    def has_web(self) -> bool:
        return self.web is not None

    @property
    def has_vision(self) -> bool:
        return self.vision is not None

    @property
    def has_code(self) -> bool:
        return self.code is not None

    @property
    def has_tools(self) -> bool:
        return self.tools is not None

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning is not None

    def available_sources(self) -> list[str]:
        """
        Returns a list of populated evidence sections.
        """

        sources: list[str] = []

        if self.has_repository:
            sources.append("repository")

        if self.has_web:
            sources.append("web")

        if self.has_vision:
            sources.append("vision")

        if self.has_code:
            sources.append("code")

        if self.has_tools:
            sources.append("tools")

        if self.has_reasoning:
            sources.append("reasoning")

        return sources

    def populated(self) -> dict[str, Any]:
        sections = {
            "repository": self.repository,
            "web": self.web,
            "vision": self.vision,
            "code": self.code,
            "tools": self.tools,
            "reasoning": self.reasoning,
        }
        return {
            name: evidence.model_dump(exclude_none=True)
            for name, evidence in sections.items()
            if evidence is not None
        }

    def model_context(self) -> dict[str, Any]:
        return self.populated()
