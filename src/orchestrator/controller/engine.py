from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from orchestrator.streaming.publisher import StreamPublisher

from ..clients.ollama import OllamaClient
from ..common.enums import (
    ChatRole,
    ControllerAction,
    SpecialistType,
)
from ..models.chat import ChatMessage
from ..models.knowledge import KnowledgeRetrieveResponse
from ..models.ollama import ModelGenerationResponse, extract_assistant_text, normalize_generation_response
from ..context.builder import (
    build_controller_messages,
    build_finalize_context,
    build_finalizer_messages,
    last_user_text,
    estimate_text_tokens,
    render_request_context,
    render_structured_context,
)
from ..logging import get_logger
from ..models.manager import ModelManager
from ..schemas import (
    ControllerPlan,
    ControllerValidation,
    CoderResult,
    NormalizedRequest,
    RoutingHints,
    ToolResult,
)
from ..settings import Settings
from .shared import current_executed_steps, plan_to_route
from .prompts import (
    build_controller_final_prompt,
    build_controller_plan_prompt,
    build_controller_validation_prompt,
    build_reasoning_prompt,
)


logger = get_logger(__name__)

_REPOSITORY_TOKENS = (
    "repository",
    "repo",
    "codebase",
    "project",
    "my orchestrator",
    "my knowledge service",
    "knowledge service",
    "homelab",
    "proxmox",
    "docker compose",
    "compose",
    "config",
    "configuration",
    "docs",
    "documentation",
    "history",
    "implementation",
)
_CODE_TOKENS = (
    "write code",
    "implement",
    "refactor",
    "debug",
    "fix ",
    "review code",
    "generate code",
    "code ",
    "function",
    "class ",
    "python",
    "typescript",
    "javascript",
)
_REASONING_TOKENS = (
    "architecture",
    "synthesize",
    "multi-document",
    "tradeoff",
    "trade-off",
    "deep reasoning",
    "design plan",
)


def _coerce_knowledge(value: Any) -> KnowledgeRetrieveResponse | None:
    if value is None:
        return None
    if isinstance(value, KnowledgeRetrieveResponse):
        return value
    if isinstance(value, dict):
        try:
            return KnowledgeRetrieveResponse.model_validate(value)
        except Exception:
            return None
    return None


def _coerce_plan(value: Any) -> ControllerPlan | None:
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


def _coerce_validation(value: Any) -> ControllerValidation | None:
    if value is None:
        return None
    if isinstance(value, ControllerValidation):
        return value
    if isinstance(value, dict):
        try:
            return ControllerValidation.model_validate(value)
        except Exception:
            return None
    return None


def _coerce_coder(value: Any) -> CoderResult | None:
    if value is None:
        return None
    if isinstance(value, CoderResult):
        return value
    if isinstance(value, dict):
        try:
            return CoderResult.model_validate(value)
        except Exception:
            return None
    return None


def _coerce_tool(value: Any) -> ToolResult | None:
    if value is None:
        return None
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, dict):
        try:
            return ToolResult.model_validate(value)
        except Exception:
            return None
    return None


def _coerce_request(value: Any) -> NormalizedRequest | None:
    if value is None:
        return None
    if isinstance(value, NormalizedRequest):
        return value
    if isinstance(value, dict):
        try:
            return NormalizedRequest.model_validate(value)
        except Exception:
            return None
    return None


def _coerce_routing_hints(value: Any) -> RoutingHints:
    if isinstance(value, RoutingHints):
        return value
    if isinstance(value, dict):
        try:
            return RoutingHints.model_validate(value)
        except Exception:
            return RoutingHints()
    request = _coerce_request(value)
    if request is not None:
        return request.routing_hints
    return RoutingHints()


def _routing_hints_from_state(state: dict[str, Any]) -> RoutingHints:
    request = _coerce_request(state.get("normalized_request"))
    if request is not None:
        return request.routing_hints
    return _coerce_routing_hints(state.get("routing_hints"))


def _request_profile(state: dict[str, Any]) -> dict[str, Any]:
    request = _coerce_request(state.get("normalized_request"))
    hints = _routing_hints_from_state(state)
    metadata = dict((request.metadata if request else state.get("metadata", {})) or {})
    text = (request.user_query if request else last_user_text(state.get("messages", []))) or ""
    lower = text.lower()
    attachments = list(request.attachments if request else state.get("attachments", []) or [])
    def attachment_type(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("attachment_type") or item.get("type") or "").lower()
        return str(getattr(item, "attachment_type", "") or "").lower()

    has_images = bool(metadata.get("has_images", False)) or any(attachment_type(item) == "image" for item in attachments)
    has_files = bool(metadata.get("has_files", False)) or any(
        attachment_type(item) not in {"", "image"} for item in attachments
    )
    repository_request = hints.repository_likelihood >= 0.55 or any(token in lower for token in _REPOSITORY_TOKENS)
    code_request = hints.code_likelihood >= 0.55 or any(token in lower for token in _CODE_TOKENS)
    reasoning_request = any(token in lower for token in _REASONING_TOKENS)
    vision_request = has_images or has_files or hints.vision_likelihood >= 0.45
    if vision_request:
        classification = "VISION"
    elif repository_request:
        classification = "KNOWLEDGE"
    elif code_request:
        classification = "CODE"
    elif reasoning_request:
        classification = "REASONING"
    else:
        classification = "GENERAL"
    return {
        "request": request,
        "metadata": metadata,
        "text": text,
        "hints": hints,
        "has_images": has_images,
        "has_files": has_files,
        "repository_request": repository_request,
        "code_request": code_request,
        "reasoning_request": reasoning_request,
        "vision_request": vision_request,
        "classification": classification,
    }


def _specialist_allowed_for_profile(profile: dict[str, Any], specialist: SpecialistType | None) -> bool:
    if specialist is None:
        return False
    if specialist == SpecialistType.VISION:
        return bool(profile.get("vision_request"))
    if specialist == SpecialistType.KNOWLEDGE:
        return bool(profile.get("repository_request"))
    if specialist == SpecialistType.CODER:
        return bool(profile.get("code_request"))
    if specialist == SpecialistType.REASONING:
        return bool(profile.get("reasoning_request") or profile.get("repository_request") or profile.get("code_request"))
    if specialist == SpecialistType.CLARIFY:
        return True
    if specialist == SpecialistType.TOOLS:
        return False
    return False


def _profile_next_specialist(profile: dict[str, Any]) -> SpecialistType | None:
    classification = str(profile.get("classification") or "GENERAL")
    if classification == "VISION":
        return SpecialistType.VISION
    if classification == "KNOWLEDGE":
        return SpecialistType.KNOWLEDGE
    if classification == "CODE":
        return SpecialistType.CODER
    if classification == "REASONING":
        return SpecialistType.REASONING
    return None


def _answer_quality(confidence: float, *, sufficient: bool, hit_count: int | None = None) -> str:
    if sufficient and confidence >= 0.8:
        return "high"
    if sufficient or confidence >= 0.5:
        return "medium"
    if hit_count == 0:
        return "low"
    return "low"


def _status_from_sufficient(sufficient: bool) -> str:
    return "success" if sufficient else "failed"


def _plan_summary_for_profile(profile: dict[str, Any]) -> str:
    classification = str(profile.get("classification") or "GENERAL")
    text = str(profile.get("text") or "").strip()
    if classification == "VISION":
        return "Use Vision because the request includes attachments that require visual understanding."
    if classification == "KNOWLEDGE":
        return "Use Knowledge first because the request is repository, project, or codebase specific."
    if classification == "CODE":
        return "Use Coder because the request asks for code generation or code work."
    if classification == "REASONING":
        return "Use Reasoning because the request explicitly needs deeper synthesis."
    if classification == "GENERAL":
        return "Answer directly from general knowledge."
    if text:
        return text[:280]
    return "Answer directly."


def _validation_summary_for_state(
    *,
    last_step: SpecialistType | None,
    evidence: dict[str, Any],
    fallback_to_general: bool,
    retry: bool,
    next_specialist: SpecialistType | None,
) -> str:
    if retry and last_step is not None:
        return f"Retrying {last_step.value} after a failed execution."
    if fallback_to_general:
        return "Knowledge was insufficient for a common-knowledge question, so the controller will answer directly."
    if next_specialist is not None:
        return f"Continue with {next_specialist.value} based on explicit specialist evidence."
    if evidence.get("sufficient"):
        return f"{last_step.value if last_step else 'Specialist'} evidence is sufficient; finalize the response."
    return "Finalize the response."


def _recommended_next_for_profile(
    profile: dict[str, Any],
    *,
    last_step: SpecialistType | None,
    evidence_sufficient: bool,
) -> SpecialistType | None:
    if evidence_sufficient:
        if last_step == SpecialistType.KNOWLEDGE:
            if profile.get("code_request"):
                return SpecialistType.CODER
            if profile.get("reasoning_request"):
                return SpecialistType.REASONING
        if last_step == SpecialistType.CODER and profile.get("repository_request") and profile.get("reasoning_request"):
            return SpecialistType.REASONING
        return None
    if not profile.get("repository_request") and not profile.get("code_request") and not profile.get("vision_request"):
        return None
    if last_step == SpecialistType.KNOWLEDGE:
        if profile.get("code_request"):
            return SpecialistType.CODER
        if profile.get("repository_request"):
            return SpecialistType.REASONING if profile.get("reasoning_request") else None
    if last_step == SpecialistType.CODER:
        if profile.get("repository_request") and not profile.get("code_request"):
            return SpecialistType.KNOWLEDGE
        if profile.get("repository_request") and profile.get("reasoning_request"):
            return SpecialistType.REASONING
    if last_step == SpecialistType.VISION:
        return None
    return None


def _coerce_generation(value: Any) -> ModelGenerationResponse | None:
    if value is None:
        return None
    if isinstance(value, ModelGenerationResponse):
        return value
    if isinstance(value, dict):
        try:
            return ModelGenerationResponse.model_validate(value)
        except Exception:
            return None
    return None


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        return {}

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    candidate = text

    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]

    try:
        import json
        return json.loads(candidate)
    except Exception:
        return {}


def _unique_steps(steps: Iterable[SpecialistType]) -> list[SpecialistType]:
    seen: set[str] = set()
    out: list[SpecialistType] = []
    for step in steps:
        key = step.value
        if key not in seen:
            seen.add(key)
            out.append(step)
    return out


def _bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_controller_step(value: Any) -> SpecialistType | None:
    if not value:
        return None
    try:
        return SpecialistType(str(value))
    except Exception:
        return None


def _fallback_final_answer(state: dict[str, Any]) -> str:
    latest_user_message = last_user_text(state.get("messages", []))
    validation = _coerce_validation(state.get("controller_validation"))
    plan = _coerce_plan(state.get("controller_plan"))
    knowledge = _coerce_knowledge(state.get("knowledge_result"))

    if validation and validation.fallback_to_general and latest_user_message:
        return (
            "I can answer this from general knowledge, but the final model returned no text. "
            f"Please retry the request: {latest_user_message}"
        )

    if knowledge and not knowledge.primary_hits:
        return (
            "I could not find useful indexed knowledge for this request. "
            "Please provide more project-specific detail, or ask me to answer "
            "from general knowledge."
        )

    if plan and plan.next_specialist == SpecialistType.CLARIFY:
        return plan.clarification_question or "I need one more detail before I can answer."

    return (
        "I could not generate a complete answer for that request. "
        "Please try again with a little more detail."
    )


def _specialist_evidence_summary(
    state: dict[str, Any],
    *,
    last_step: SpecialistType | None,
    settings: Settings,
) -> dict[str, Any]:
    profile = _request_profile(state)
    step = last_step.value if last_step else "unknown"
    summary: dict[str, Any] = {
        "specialist_type": step,
        "status": "unknown",
        "confidence": 0.0,
        "result_summary": "",
        "hit_count": None,
        "answer_quality": "low",
        "needs_additional_specialist": False,
        "recommended_next_specialist": None,
        "sufficient": False,
    }

    if last_step == SpecialistType.KNOWLEDGE:
        knowledge = _coerce_knowledge(state.get("knowledge_result"))
        if knowledge is None:
            summary.update(
                {
                    "status": "failed",
                    "result_summary": "Knowledge retrieval did not return a result.",
                    "hit_count": 0,
                }
            )
            return summary

        hit_count = len(knowledge.primary_hits or [])
        sufficient = bool(
            knowledge.context
            and hit_count >= settings.knowledge_min_hits
            and knowledge.confidence >= settings.knowledge_min_score
        )
        recommended_next = _recommended_next_for_profile(profile, last_step=last_step, evidence_sufficient=sufficient)
        summary.update(
            {
                "status": _status_from_sufficient(sufficient),
                "confidence": knowledge.confidence,
                "result_summary": knowledge.retrieval_reason or (knowledge.context or "")[:400],
                "hit_count": hit_count,
                "answer_quality": _answer_quality(knowledge.confidence, sufficient=sufficient, hit_count=hit_count),
                "needs_additional_specialist": bool(recommended_next),
                "recommended_next_specialist": recommended_next.value if recommended_next else None,
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.CODER:
        coder = _coerce_coder(state.get("coder_result"))
        if coder is None:
            summary.update(
                {
                    "status": "failed",
                    "result_summary": "Coder returned no result.",
                }
            )
            return summary
        sufficient = bool((coder.summary or coder.code or "").strip())
        recommended_next = _recommended_next_for_profile(profile, last_step=last_step, evidence_sufficient=sufficient)
        summary.update(
            {
                "status": _status_from_sufficient(sufficient),
                "confidence": coder.confidence,
                "result_summary": coder.summary or coder.code[:400],
                "answer_quality": _answer_quality(coder.confidence, sufficient=sufficient),
                "needs_additional_specialist": bool(recommended_next),
                "recommended_next_specialist": recommended_next.value if recommended_next else None,
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.TOOLS:
        tool = _coerce_tool(state.get("tool_result"))
        if tool is None:
            summary.update(
                {
                    "status": "failed",
                    "result_summary": "Tool step returned no result.",
                }
            )
            return summary
        sufficient = tool.status == "ok" and bool((tool.summary or tool.result or tool.raw_text))
        summary.update(
            {
                "status": tool.status,
                "confidence": 1.0 if sufficient else 0.0,
                "result_summary": tool.summary or str(tool.result)[:400],
                "answer_quality": _answer_quality(1.0 if sufficient else 0.0, sufficient=sufficient),
                "needs_additional_specialist": False,
                "recommended_next_specialist": None,
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.VISION:
        vision_context = str(state.get("vision_context", "") or "").strip()
        summary.update(
            {
                "status": _status_from_sufficient(bool(vision_context)),
                "confidence": 1.0 if vision_context else 0.0,
                "result_summary": vision_context[:400],
                "answer_quality": _answer_quality(1.0 if vision_context else 0.0, sufficient=bool(vision_context)),
                "needs_additional_specialist": False,
                "recommended_next_specialist": None,
                "sufficient": bool(vision_context),
            }
        )
        return summary

    return summary


@dataclass(slots=True)
class ControllerEngine:
    settings: Settings
    ollama: OllamaClient
    models: ModelManager

    def _state_messages(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        controller_messages = state.get("controller_messages")
        if controller_messages:
            return list(controller_messages or [])
        return list(state.get("messages", []) or [])

    def _request_context(self, state: dict[str, Any]) -> str:
        normalized = _coerce_request(state.get("normalized_request"))
        if normalized is not None:
            return render_request_context(normalized)
        metadata = state.get("metadata", {}) or {}
        hints = _coerce_routing_hints(state.get("routing_hints"))
        if not metadata and not hints:
            return ""
        return render_request_context(
            {
                "original_messages": [],
                "controller_messages": [],
                "user_query": str(state.get("user_text", "") or ""),
                "metadata": metadata,
                "routing_hints": hints.model_dump(exclude_none=True),
                "attachments": state.get("attachments", []) or [],
            }
        )

    def _structured_state_prompt(self, state: dict[str, Any]) -> str:
        plan = _coerce_plan(state.get("execution_plan") or state.get("controller_plan"))
        validation = _coerce_validation(state.get("execution_plan") or state.get("controller_validation"))
        knowledge_result = state.get("knowledge_result")
        vision_context = state.get("vision_context", "")
        coder_result = state.get("coder_result")
        tool_result = state.get("tool_result")
        reasoning_result = state.get("reasoning_result")
        request_context = self._request_context(state)
        request_context_block = f"{request_context}\n\n" if request_context else ""
        return request_context_block + render_structured_context(
            vision_context=vision_context,
            knowledge_result=_coerce_knowledge(knowledge_result),
            coder_result=_coerce_coder(coder_result),
            tool_result=_coerce_tool(tool_result),
            reasoning_result=_coerce_generation(reasoning_result),
            controller_plan=_coerce_plan(plan),
            controller_validation=_coerce_validation(validation),
        )

    async def plan(self, state: dict[str, Any]) -> ControllerPlan:
        profile = _request_profile(state)

        messages = build_controller_messages(
            system_prompt=build_controller_plan_prompt(),
            messages=self._state_messages(state),
            request_context=self._request_context(state),
        )

        response = await self.ollama.chat(
            model=self.models.controller().name,
            messages=messages,
            temperature=self.settings.controller_plan_temperature,
            max_tokens=self.settings.controller_plan_max_tokens,
            stream=False,
            keep_alive=self.settings.controller_keep_alive,
        )

        parsed = _extract_json_object(response.content)
        parsed_classification = str(
            parsed.get("classification")
            or parsed.get("route")
            or parsed.get("category")
            or ""
        ).strip().upper()
        deterministic_next = _profile_next_specialist(profile)
        classification = str(profile.get("classification") or "GENERAL")
        explicit_next = deterministic_next
        complete = explicit_next is None
        needs_reasoning = explicit_next == SpecialistType.REASONING
        requires_clarification = False
        pending_specialists = [explicit_next] if explicit_next is not None else []

        if classification == "GENERAL":
            complete = True
            explicit_next = None
            pending_specialists = []
            needs_reasoning = False
        elif explicit_next is None:
            complete = True

        rejected_reasons: list[str] = []
        if parsed_classification and parsed_classification not in {
            "GENERAL",
            "KNOWLEDGE",
            "CODE",
            "VISION",
            "TOOLS",
            "REASONING",
            "CLARIFY",
        }:
            rejected_reasons.append(f"invalid_classification={parsed_classification}")
        elif parsed_classification and parsed_classification != classification:
            rejected_reasons.append(f"model_classification={parsed_classification}")
        parsed_next = _normalize_controller_step(parsed.get("next_specialist") or parsed.get("next_step"))
        if parsed_next is not None and parsed_next != explicit_next:
            rejected_reasons.append(f"model_next={parsed_next.value}")
        if parsed.get("pending_specialists"):
            rejected_reasons.append("model_pending_specialists_ignored")

        plan = ControllerPlan(
            classification=classification,
            intent=f"{classification.lower()} request",
            summary=_plan_summary_for_profile(profile),
            complexity="medium",
            confidence=_safe_float(parsed.get("confidence"), 0.0),
            action=ControllerAction.FINALIZE if complete else ControllerAction.CONTINUE,
            complete=complete,
            next_specialist=explicit_next,
            pending_specialists=_unique_steps(pending_specialists),
            retry=False,
            retry_reason="",
            needs_reasoning=needs_reasoning,
            final_answer_ready=complete,
            clarification_question=None,
            fallback_to_general=classification == "GENERAL",
            knowledge_sufficient=None,
            completion_condition="Finalize when the planned specialist evidence is sufficient.",
            explanation=_plan_summary_for_profile(profile),
        )

        if plan.classification == "GENERAL":
            plan.next_specialist = None
            plan.pending_specialists = []
            plan.needs_reasoning = False
            plan.complete = True
            plan.final_answer_ready = True
            plan.action = ControllerAction.FINALIZE
            plan.fallback_to_general = True

        plan.route_hint = plan_to_route(plan)

        logger.debug(
            "controller_plan_decision %s",
            json.dumps(
                {
                    "classification": plan.classification,
                    "selected_next_specialist": plan.next_specialist.value if plan.next_specialist else None,
                    "complete": plan.complete,
                    "request_evidence": {
                        "has_images": profile.get("has_images", False),
                        "has_files": profile.get("has_files", False),
                        "repository_request": profile.get("repository_request", False),
                        "code_request": profile.get("code_request", False),
                        "reasoning_request": profile.get("reasoning_request", False),
                        "vision_request": profile.get("vision_request", False),
                    },
                    "rejected_model_signals": rejected_reasons,
                },
                sort_keys=True,
                default=str,
            ),
        )
        return plan

    async def validate(
        self,
        state: dict[str, Any],
        *,
        last_step: SpecialistType | None = None,
    ) -> ControllerValidation:
        step_text = last_step.value if last_step else "unknown"
        state_summary = self._structured_state_prompt(state)
        evidence = _specialist_evidence_summary(state, last_step=last_step, settings=self.settings)
        profile = _request_profile(state)

        validation_messages = build_controller_messages(
            system_prompt=build_controller_validation_prompt(),
            messages=self._state_messages(state),
            request_context=self._request_context(state),
            structured_context=(
                "Current state summary:\n\n"
                f"{state_summary or 'No structured context yet.'}"
            ),
            additional_context=(
                f"Last specialist step: {step_text}\n\n"
                f"Specialist evidence summary:\n{evidence}"
            ),
        )

        response = await self.ollama.chat(
            model=self.models.controller().name,
            messages=validation_messages,
            temperature=self.settings.controller_validate_temperature,
            max_tokens=self.settings.controller_validate_max_tokens,
            stream=False,
            keep_alive=self.settings.controller_keep_alive,
        )

        parsed = _extract_json_object(response.content)
        plan = _coerce_plan(state.get("execution_plan") or state.get("controller_plan"))
        request_classification = str(profile.get("classification") or "GENERAL")
        confidence = _safe_float(parsed.get("confidence"), 0.0)
        knowledge_sufficient: bool | None = evidence.get("sufficient") if last_step == SpecialistType.KNOWLEDGE else None
        fallback_to_general = False
        retry = False
        needs_reasoning = False
        requires_clarification = False
        complete = True
        action = ControllerAction.FINALIZE
        next_specialist: SpecialistType | None = None

        current_status = str(evidence.get("status") or "").strip().lower()
        candidate = _normalize_controller_step(evidence.get("recommended_next_specialist"))
        candidate_allowed = _specialist_allowed_for_profile(profile, candidate)
        candidate_is_new = candidate is not None and candidate.value not in current_executed_steps(state)
        needs_additional_specialist = bool(evidence.get("needs_additional_specialist", False))

        retry_counts: dict[str, int] = {}
        for key, value in (state.get("retry_counts", {}) or {}).items():
            try:
                retry_counts[str(key)] = int(value)
            except Exception:
                retry_counts[str(key)] = 0

        if last_step is not None and current_status == "failed" and retry_counts.get(last_step.value, 0) < self.settings.max_specialist_retries:
            retry = True
            next_specialist = last_step
            action = ControllerAction.CONTINUE
            complete = False
            retry_reason = "specialist failed; retrying once"
        elif last_step == SpecialistType.KNOWLEDGE:
            sufficient = bool(evidence.get("sufficient", False))
            knowledge_sufficient = sufficient
            if needs_additional_specialist and candidate_allowed and candidate_is_new:
                next_specialist = candidate
                complete = False
                if candidate == SpecialistType.REASONING:
                    action = ControllerAction.REASON
                    needs_reasoning = True
                elif candidate == SpecialistType.CLARIFY:
                    action = ControllerAction.CLARIFY
                    requires_clarification = True
                else:
                    action = ControllerAction.CONTINUE
            elif not sufficient and request_classification == "GENERAL":
                fallback_to_general = True
            elif candidate is not None and not candidate_allowed:
                fallback_to_general = request_classification == "GENERAL"
        elif last_step == SpecialistType.CODER:
            sufficient = bool(evidence.get("sufficient", False))
            if needs_additional_specialist and candidate_allowed and candidate_is_new:
                next_specialist = candidate
                complete = False
                if candidate == SpecialistType.REASONING:
                    action = ControllerAction.REASON
                    needs_reasoning = True
                elif candidate == SpecialistType.CLARIFY:
                    action = ControllerAction.CLARIFY
                    requires_clarification = True
                else:
                    action = ControllerAction.CONTINUE
            elif not sufficient and request_classification == "GENERAL":
                fallback_to_general = True
        elif last_step == SpecialistType.VISION:
            sufficient = bool(evidence.get("sufficient", False))
            if needs_additional_specialist and candidate_allowed and candidate_is_new:
                next_specialist = candidate
                complete = False
                action = ControllerAction.CONTINUE if candidate not in {SpecialistType.REASONING, SpecialistType.CLARIFY} else (
                    ControllerAction.REASON if candidate == SpecialistType.REASONING else ControllerAction.CLARIFY
                )
                needs_reasoning = candidate == SpecialistType.REASONING
                requires_clarification = candidate == SpecialistType.CLARIFY
        elif candidate_allowed and candidate_is_new:
            next_specialist = candidate
            complete = False
            if candidate == SpecialistType.REASONING:
                action = ControllerAction.REASON
                needs_reasoning = True
            elif candidate == SpecialistType.CLARIFY:
                action = ControllerAction.CLARIFY
                requires_clarification = True
            else:
                action = ControllerAction.CONTINUE

        if next_specialist is None and not fallback_to_general and not retry and not needs_reasoning and not requires_clarification:
            complete = True

        if fallback_to_general:
            action = ControllerAction.FINALIZE
            next_specialist = None
            complete = True
            needs_reasoning = False
            requires_clarification = False

        if candidate is not None and not candidate_allowed:
            logger.debug(
                "controller_validation_rejected_candidate %s",
                json.dumps(
                    {
                        "last_step": last_step.value if last_step else None,
                        "candidate": candidate.value,
                        "reason": "candidate not justified by request evidence",
                        "request_classification": request_classification,
                    },
                    sort_keys=True,
                    default=str,
                ),
            )

        summary = _validation_summary_for_state(
            last_step=last_step,
            evidence=evidence,
            fallback_to_general=fallback_to_general,
            retry=retry,
            next_specialist=next_specialist,
        )
        retry_reason = summary if retry else ""

        executed_steps = {
            str(item)
            for item in (state.get("executed_specialists", []) or [])
            if str(item).strip()
        }
        failed_steps = {
            str(item)
            for item in (state.get("failed_specialists", []) or [])
            if str(item).strip()
        }
        if next_specialist is not None and not retry:
            if (
                next_specialist.value in executed_steps
                and next_specialist.value not in failed_steps
            ):
                next_specialist = None
                if needs_reasoning:
                    action = ControllerAction.REASON
                    next_specialist = SpecialistType.REASONING
                elif requires_clarification:
                    action = ControllerAction.CLARIFY
                    next_specialist = SpecialistType.CLARIFY
                else:
                    action = ControllerAction.FINALIZE
                    complete = True
            elif (
                next_specialist.value in failed_steps
                and retry_counts.get(next_specialist.value, 0.0) >= float(self.settings.max_specialist_retries)
            ):
                next_specialist = None
                if needs_reasoning:
                    action = ControllerAction.REASON
                    next_specialist = SpecialistType.REASONING
                elif requires_clarification:
                    action = ControllerAction.CLARIFY
                    next_specialist = SpecialistType.CLARIFY
                else:
                    action = ControllerAction.FINALIZE
                    complete = True

        pending_specialists = [next_specialist] if next_specialist is not None and not complete else []
        summary = summary or ("Retrying specialist after failure." if retry else "Validation complete.")

        validation = ControllerValidation(
            action=action,
            summary=summary,
            confidence=confidence,
            complete=complete or action == ControllerAction.FINALIZE,
            next_specialist=next_specialist,
            pending_specialists=_unique_steps(pending_specialists),
            retry=retry,
            retry_reason=retry_reason,
            needs_reasoning=needs_reasoning,
            final_answer_ready=complete or action == ControllerAction.FINALIZE,
            fallback_to_general=fallback_to_general,
            knowledge_sufficient=knowledge_sufficient,
            reason=summary,
            issues=[],
            notes="",
            classification=(plan.classification if plan else "GENERAL"),
        )

        logger.debug(
            "controller_validation_decision %s",
            json.dumps(
                {
                    "last_step": step_text,
                    "action": validation.action.value,
                    "selected_next_specialist": validation.next_specialist.value if validation.next_specialist else None,
                    "retry": validation.retry,
                    "fallback_to_general": validation.fallback_to_general,
                    "evidence": evidence,
                    "request_classification": request_classification,
                },
                sort_keys=True,
                default=str,
            ),
        )
        return validation
    
    async def finalize(
                self,
                state: dict[str, Any],
                publisher: StreamPublisher | None = None,
            ) -> ModelGenerationResponse:
        """
        Produce a final answer with the resident controller.
        If specialist evidence exists, synthesize it into a user-facing answer.
        """
        finalizer_context = build_finalize_context(state)
        context_json = json.dumps(finalizer_context, separators=(",", ":"), ensure_ascii=False).strip()
        finalizer_prompt = build_controller_final_prompt()

        messages = build_finalizer_messages(
            system_prompt=finalizer_prompt,
            messages=self._state_messages(state),
            evidence_context=context_json or '{"question":"","sources":[]}',
        )

        prompt_tokens = estimate_text_tokens(finalizer_prompt)
        context_tokens = estimate_text_tokens(context_json)
        logger.debug(
            "finalizer_context %s",
            json.dumps(
                {
                    "finalizer_prompt_tokens": prompt_tokens,
                    "finalizer_context_tokens": context_tokens,
                    "finalizer_sources": len(finalizer_context.get("sources", []) or []),
                },
                sort_keys=True,
                default=str,
            ),
        )

        model_name = self.models.controller().name

        try:
            if publisher is not None:
                await publisher.llm_started(model=model_name)

                content_parts: list[str] = []
                final_raw: dict[str, Any] = {}

                async for chunk in self.ollama.stream_chat(
                    model=model_name,
                    messages=messages,
                    temperature=self.settings.controller_finalize_temperature,
                    max_tokens=self.settings.controller_finalize_max_tokens,
                    keep_alive=self.settings.controller_keep_alive,
                ):
                    if chunk.content:
                        content_parts.append(chunk.content)
                        await publisher.llm_token(chunk.content)
                    final_raw = chunk.raw or final_raw

                extracted = extract_assistant_text("".join(content_parts)) or extract_assistant_text(final_raw)
                if not extracted.strip():
                    extracted = _fallback_final_answer(state)

                await publisher.llm_finished()

                return ModelGenerationResponse(
                    model=model_name,
                    content=extracted.strip(),
                    raw=final_raw,
                )

            response = await self.ollama.chat(
                model=model_name,
                messages=messages,
                temperature=self.settings.controller_finalize_temperature,
                max_tokens=self.settings.controller_finalize_max_tokens,
                stream=False,
                keep_alive=self.settings.controller_keep_alive,
            )

            extracted = extract_assistant_text(response.content) or extract_assistant_text(response.raw)
            if not extracted.strip():
                return ModelGenerationResponse(
                    model=response.model,
                    content=_fallback_final_answer(state),
                    raw=response.raw,
                )

            return ModelGenerationResponse(
                model=response.model,
                content=extracted.strip(),
                raw=response.raw,
            )

        except Exception as exc:
            if publisher is not None:
                await publisher.error(str(exc), stage="finalize")
            raise

    async def reason(self, state: dict[str, Any]) -> ModelGenerationResponse:
        latest_user_message = last_user_text(state.get("messages", []))
        structured_context = self._structured_state_prompt(state)

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=build_reasoning_prompt()),
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "structured_context"},
                content=structured_context or "No structured context available.",
            ),
        ]
        if latest_user_message:
            messages.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))

        response = await self.ollama.chat(
            model=self.models.reasoning().name,
            messages=messages,
            temperature=self.settings.reasoning_temperature,
            max_tokens=self.settings.reasoning_max_tokens,
            stream=False,
            keep_alive=self.settings.reasoning_keep_alive,
            think=self.settings.reasoning_think,
        )
        return normalize_generation_response(response.model, response.content or response.raw)
