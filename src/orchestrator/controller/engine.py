from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from orchestrator.streaming.publisher import StreamPublisher

from ..clients.ollama import OllamaClient
from ..common.enums import ChatRole, ControllerAction, SpecialistType
from ..common.utils import _extract_json_object
from ..context.builder import (
    build_controller_messages,
    build_finalize_context,
    build_finalizer_messages,
    estimate_text_tokens,
    last_user_text,
    render_request_context,
)
from ..logging import get_logger
from ..models.chat import ChatMessage
from ..models.knowledge import KnowledgeRetrieveResponse
from ..models.manager import ModelManager
from ..models.ollama import (
    ModelGenerationResponse,
    extract_assistant_text,
    normalize_generation_response,
)
from ..models.web import WebSearchResult
from ..schemas import (
    CoderResult,
    ControllerPlan,
    ControllerValidation,
    NormalizedRequest,
    RoutingHints,
    ToolResult,
)
from ..settings import Settings
from .prompts import (
    build_controller_final_prompt,
    build_controller_plan_prompt,
    build_controller_validation_prompt,
    build_reasoning_prompt,
)
from .shared import current_executed_steps, plan_to_route


logger = get_logger(__name__)


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


def _coerce_web(value: Any) -> WebSearchResult | None:
    if value is None:
        return None
    if isinstance(value, WebSearchResult):
        return value
    if isinstance(value, dict):
        try:
            return WebSearchResult.model_validate(value)
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


def _bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_step(value: Any) -> SpecialistType | None:
    if value is None:
        return None
    if isinstance(value, SpecialistType):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return SpecialistType(text)
    except Exception:
        return None


def _step_name(step: Any) -> str:
    if isinstance(step, SpecialistType):
        return step.value
    return str(step or "").strip().lower()


def _unique_steps(steps: Iterable[SpecialistType]) -> list[SpecialistType]:
    seen: set[str] = set()
    out: list[SpecialistType] = []
    for step in steps:
        if step is None:
            continue
        key = step.value
        if key not in seen:
            seen.add(key)
            out.append(step)
    return out


def _request_profile(state: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize request structure only.
    Do not infer routing here.
    """
    request = _coerce_request(state.get("normalized_request"))
    hints = _routing_hints_from_state(state)
    metadata = dict((request.metadata if request else state.get("metadata", {})) or {})
    text = (request.user_query if request else last_user_text(state.get("messages", []))) or ""
    attachments = list(request.attachments if request else state.get("attachments", []) or [])

    def attachment_type(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("attachment_type") or item.get("type") or "").lower()
        return str(getattr(item, "attachment_type", "") or "").lower()

    has_images = bool(metadata.get("has_images", False)) or any(
        attachment_type(item) == "image" for item in attachments
    )
    has_files = bool(metadata.get("has_files", False)) or any(
        attachment_type(item) not in {"", "image"} for item in attachments
    )

    return {
        "request": request,
        "metadata": metadata,
        "text": text,
        "hints": hints,
        "has_images": has_images,
        "has_files": has_files,
    }


def _structured_state_prompt(state: dict[str, Any]) -> str:
    context = build_finalize_context(state)
    return json.dumps(context, indent=2, ensure_ascii=False, default=str)


def _parse_plan_flags(parsed: dict[str, Any]) -> dict[str, bool]:
    return {
        "requires_repository": _bool_from_any(
            parsed.get("requires_repository")
            or parsed.get("needs_rag")
            or parsed.get("needs_repository")
        ),
        "requires_web": _bool_from_any(
            parsed.get("requires_web")
            or parsed.get("use_web_search")
            or parsed.get("needs_web")
        ),
        "requires_vision": _bool_from_any(
            parsed.get("requires_vision")
            or parsed.get("needs_vision")
        ),
        "requires_tools": _bool_from_any(
            parsed.get("requires_tools")
            or parsed.get("needs_tools")
        ),
        "requires_code": _bool_from_any(
            parsed.get("requires_code")
            or parsed.get("needs_code")
        ),
        "requires_reasoning": _bool_from_any(
            parsed.get("requires_reasoning")
            or parsed.get("needs_reasoning")
        ),
        "requires_clarify": _bool_from_any(
            parsed.get("requires_clarify")
            or parsed.get("needs_clarification")
        ),
    }


def _plan_queue_from_json(parsed: dict[str, Any], classification: str) -> list[SpecialistType]:
    queue: list[SpecialistType] = []

    pending_raw = parsed.get("pending_specialists")
    if isinstance(pending_raw, list):
        for item in pending_raw:
            step = _normalize_step(item)
            if step is not None:
                queue.append(step)

    if not queue:
        flags = _parse_plan_flags(parsed)
        ordered_flags = [
            (SpecialistType.VISION, flags["requires_vision"]),
            (SpecialistType.KNOWLEDGE, flags["requires_repository"]),
            (SpecialistType.WEB, flags["requires_web"]),
            (SpecialistType.TOOLS, flags["requires_tools"]),
            (SpecialistType.CODER, flags["requires_code"]),
            (SpecialistType.REASONING, flags["requires_reasoning"]),
            (SpecialistType.CLARIFY, flags["requires_clarify"]),
        ]
        for step, enabled in ordered_flags:
            if enabled:
                queue.append(step)

    if not queue:
        explicit_next = _normalize_step(parsed.get("next_specialist") or parsed.get("next_step"))
        if explicit_next is not None:
            queue.append(explicit_next)

    if not queue:
        classification = str(classification or "GENERAL").strip().upper()
        if classification == "VISION":
            queue = [SpecialistType.VISION]
        elif classification == "KNOWLEDGE":
            queue = [SpecialistType.KNOWLEDGE]
        elif classification == "CODE":
            queue = [SpecialistType.CODER]
        elif classification == "REASONING":
            queue = [SpecialistType.REASONING]
        elif classification == "TOOLS":
            queue = [SpecialistType.TOOLS]
        elif classification == "CLARIFY":
            queue = [SpecialistType.CLARIFY]

    return _unique_steps(queue)


def _plan_summary_from_json(parsed: dict[str, Any], classification: str) -> str:
    flags = _parse_plan_flags(parsed)
    reasons: list[str] = []

    if flags["requires_repository"]:
        reasons.append("repository evidence")
    if flags["requires_web"]:
        reasons.append("fresh web evidence")
    if flags["requires_vision"]:
        reasons.append("vision evidence")
    if flags["requires_tools"]:
        reasons.append("tool execution")
    if flags["requires_code"]:
        reasons.append("code work")
    if flags["requires_reasoning"]:
        reasons.append("deeper synthesis")
    if flags["requires_clarify"]:
        reasons.append("clarification")

    if reasons:
        if len(reasons) == 1:
            return f"Use {reasons[0]}."
        if len(reasons) == 2:
            return f"Use {reasons[0]} and {reasons[1]}."
        return "Use " + ", ".join(reasons[:-1]) + f", and {reasons[-1]}."

    classification = str(classification or "GENERAL").strip().upper()
    if classification == "GENERAL":
        return "Answer directly from general knowledge."
    if classification == "KNOWLEDGE":
        return "Use Knowledge because the request is repository, project, or codebase specific."
    if classification == "CODE":
        return "Use Coder because the request asks for code work."
    if classification == "VISION":
        return "Use Vision because the request includes image or document attachments."
    if classification == "REASONING":
        return "Use Reasoning because the request needs deeper synthesis."
    if classification == "TOOLS":
        return "Use Tools because the request requires execution."
    return "Answer directly."


def _next_pending_specialist(plan: ControllerPlan | None, executed_steps: Iterable[str]) -> SpecialistType | None:
    if plan is None:
        return None
    executed = {_step_name(step) for step in executed_steps}
    pending = list(plan.pending_specialists or [])
    if not pending and plan.next_specialist is not None:
        pending = [plan.next_specialist]
    for step in pending:
        if step is None:
            continue
        if _step_name(step) not in executed:
            return step
    return None


def _plan_execution_queue(plan: ControllerPlan | None) -> list[SpecialistType]:
    if plan is None:
        return []
    queue = list(plan.execution_queue or [])
    if not queue:
        queue = list(plan.pending_specialists or [])
    if not queue and plan.next_specialist is not None:
        queue = [plan.next_specialist]
    return _unique_steps(queue)


def _fallback_final_answer(state: dict[str, Any]) -> str:
    latest_user_message = last_user_text(state.get("messages", []))
    validation = _coerce_validation(state.get("controller_validation"))
    plan = _coerce_plan(state.get("execution_plan") or state.get("controller_plan"))
    knowledge = _coerce_knowledge(state.get("knowledge_result"))

    if validation and validation.fallback_to_general and latest_user_message:
        return (
            "I can answer this from general knowledge, but the final model returned no text. "
            f"Please retry the request: {latest_user_message}"
        )

    if knowledge and not knowledge.primary_hits:
        return (
            "I could not find useful indexed knowledge for this request. "
            "Please provide more project-specific detail, or ask me to answer from general knowledge."
        )

    if plan and plan.next_specialist == SpecialistType.CLARIFY:
        return plan.clarification_question or "I need one more detail before I can answer."

    return "I could not generate a complete answer for that request. Please try again with a little more detail."


def _specialist_evidence_summary(
    state: dict[str, Any],
    *,
    last_step: SpecialistType | None,
    settings: Settings,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "specialist_type": last_step.value if last_step else "unknown",
        "status": "unknown",
        "confidence": 0.0,
        "result_summary": "",
        "hit_count": None,
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
        summary.update(
            {
                "status": "success" if sufficient else "failed",
                "confidence": knowledge.confidence,
                "result_summary": knowledge.retrieval_reason or (knowledge.context or "")[:400],
                "hit_count": hit_count,
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.WEB:
        web = _coerce_web(state.get("web_search_result"))
        if web is None:
            summary.update({"status": "failed", "result_summary": "Web search did not return a result.", "hit_count": 0})
            return summary

        sufficient = bool(web.results)
        summary.update(
            {
                "status": "success" if sufficient else "failed",
                "confidence": 1.0 if sufficient else 0.0,
                "result_summary": "Web search returned live evidence." if sufficient else (web.error or "No web results."),
                "hit_count": len(web.results),
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.CODER:
        coder = _coerce_coder(state.get("coder_result"))
        if coder is None:
            summary.update({"status": "failed", "result_summary": "Coder returned no result."})
            return summary
        sufficient = bool((coder.summary or coder.code or "").strip())
        summary.update(
            {
                "status": "success" if sufficient else "failed",
                "confidence": coder.confidence,
                "result_summary": coder.summary or coder.code[:400],
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.VISION:
        vision_context = str(state.get("vision_context", "") or "").strip()
        sufficient = bool(vision_context)
        summary.update(
            {
                "status": "success" if sufficient else "failed",
                "confidence": 1.0 if sufficient else 0.0,
                "result_summary": vision_context[:400],
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.TOOLS:
        tool = _coerce_tool(state.get("tool_result"))
        if tool is None:
            summary.update({"status": "failed", "result_summary": "Tool step returned no result."})
            return summary
        sufficient = tool.status == "ok" and bool((tool.summary or tool.result or tool.raw_text))
        summary.update(
            {
                "status": tool.status,
                "confidence": 1.0 if sufficient else 0.0,
                "result_summary": tool.summary or str(tool.result)[:400],
                "sufficient": sufficient,
            }
        )
        return summary

    return summary


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
        return f"Continue with {next_specialist.value} based on the current execution plan."
    if evidence.get("sufficient"):
        return f"{last_step.value if last_step else 'Specialist'} evidence is sufficient; finalize the response."
    return "Finalize the response."


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
        return _structured_state_prompt(state)

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
        classification = str(
            parsed.get("classification")
            or parsed.get("route")
            or parsed.get("category")
            or "GENERAL"
        ).strip().upper()

        if classification not in {
            "GENERAL",
            "KNOWLEDGE",
            "CODE",
            "VISION",
            "TOOLS",
            "REASONING",
            "CLARIFY",
        }:
            classification = "GENERAL"

        queue = _plan_queue_from_json(parsed, classification)
        next_specialist = queue[0] if queue else None
        use_web_search = bool(
            _bool_from_any(parsed.get("use_web_search") or parsed.get("requires_web"))
            and self.settings.web_search_enabled
        )

        if next_specialist is None and use_web_search:
            next_specialist = SpecialistType.WEB
            queue = [SpecialistType.WEB]

        complete = next_specialist is None
        if classification == "GENERAL" and not use_web_search and not queue:
            complete = True

        rejected_reasons: list[str] = []
        parsed_next = _normalize_step(parsed.get("next_specialist") or parsed.get("next_step"))
        if parsed_next is not None and parsed_next != next_specialist:
            rejected_reasons.append(f"model_next={parsed_next.value}")
        if parsed.get("pending_specialists"):
            rejected_reasons.append("model_pending_specialists_ignored")

        plan = ControllerPlan(
            classification=classification,
            intent=str(parsed.get("intent") or f"{classification.lower()} request"),
            summary=str(parsed.get("summary") or _plan_summary_from_json(parsed, classification)),
            complexity=str(parsed.get("complexity") or "medium"),
            confidence=_safe_float(parsed.get("confidence"), 0.0),
            action=ControllerAction.FINALIZE if complete else ControllerAction.CONTINUE,
            complete=complete,
            next_specialist=next_specialist,
            pending_specialists=_unique_steps(queue),
            retry=False,
            retry_reason="",
            needs_reasoning=bool(parsed.get("needs_reasoning") or parsed.get("requires_reasoning")),
            final_answer_ready=complete,
            clarification_question=None,
            fallback_to_general=classification == "GENERAL" and not use_web_search and not queue,
            knowledge_sufficient=None,
            completion_condition="Finalize when all planned specialist evidence is sufficient.",
            explanation=str(parsed.get("explanation") or _plan_summary_from_json(parsed, classification)),
            use_web_search=use_web_search,
        )

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
                        "use_web_search": plan.use_web_search,
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
        plan = _coerce_plan(state.get("execution_plan") or state.get("controller_plan"))

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
                f"Specialist evidence summary:\n{json.dumps(evidence, sort_keys=True, default=str)}"
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
        confidence = _safe_float(parsed.get("confidence"), 0.0)

        retry_counts: dict[str, int] = {}
        for key, value in (state.get("retry_counts", {}) or {}).items():
            try:
                retry_counts[str(key)] = int(value)
            except Exception:
                retry_counts[str(key)] = 0

        executed = {_step_name(step) for step in current_executed_steps(state)}
        planned_queue = _plan_execution_queue(plan)
        remaining_queue = [
            step for step in planned_queue
            if _step_name(step) not in executed
        ]

        retry = False
        fallback_to_general = False
        needs_reasoning = False
        requires_clarification = False
        current_status = str(evidence.get("status") or "").strip().lower()

        if last_step is not None and current_status == "failed":
            retries_used = retry_counts.get(last_step.value, 0)
            if retries_used < self.settings.max_specialist_retries:
                retry = True

        if retry and last_step is not None:
            remaining_queue = [last_step] + [
                step for step in remaining_queue
                if _step_name(step) != last_step.value
            ]

        next_specialist = remaining_queue[0] if remaining_queue else None
        complete = not remaining_queue
        action = ControllerAction.FINALIZE if complete else ControllerAction.CONTINUE

        if not retry and next_specialist is not None:
            if next_specialist == SpecialistType.REASONING:
                action = ControllerAction.REASON
                needs_reasoning = True
            elif next_specialist == SpecialistType.CLARIFY:
                action = ControllerAction.CLARIFY
                requires_clarification = True
            else:
                action = ControllerAction.CONTINUE

        if not retry and next_specialist is None and not evidence.get("sufficient") and (
            plan is not None and plan.classification == "GENERAL" and not bool(plan.use_web_search)
        ):
            fallback_to_general = True
            action = ControllerAction.FINALIZE
            complete = True

        summary = _validation_summary_for_state(
            last_step=last_step,
            evidence=evidence,
            fallback_to_general=fallback_to_general,
            retry=retry,
            next_specialist=next_specialist,
        )
        retry_reason = summary if retry else ""

        validation = ControllerValidation(
            action=action,
            summary=summary,
            confidence=confidence,
            complete=complete or action == ControllerAction.FINALIZE,
            next_specialist=next_specialist,
            pending_specialists=_unique_steps(remaining_queue),
            execution_queue=_unique_steps(remaining_queue),
            retry=retry,
            retry_reason=retry_reason,
            needs_reasoning=needs_reasoning,
            final_answer_ready=complete or action == ControllerAction.FINALIZE,
            fallback_to_general=fallback_to_general,
            knowledge_sufficient=(evidence.get("sufficient") if last_step == SpecialistType.KNOWLEDGE else None),
            reason=summary,
            issues=[],
            notes="",
            classification=(plan.classification if plan else "GENERAL"),
            use_web_search=bool(plan.use_web_search if plan else False),
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
                    "pending_plan": [
                        step.value if isinstance(step, SpecialistType) else str(step)
                        for step in (plan.pending_specialists if plan else [])
                    ],
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
        finalizer_context = build_finalize_context(state)
        context_json = json.dumps(finalizer_context, separators=(",", ":"), ensure_ascii=False).strip()
        finalizer_prompt = build_controller_final_prompt()

        messages = build_finalizer_messages(
            system_prompt=finalizer_prompt,
            messages=self._state_messages(state),
            evidence_context=context_json or '{"question":"","sources":[]}',
        )

        logger.debug(
            "finalizer_context %s",
            json.dumps(
                {
                    "finalizer_prompt_tokens": estimate_text_tokens(finalizer_prompt),
                    "finalizer_context_tokens": estimate_text_tokens(context_json),
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