from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..clients.ollama import OllamaClient
from ..common.enums import (
    ChatRole,
    ControllerAction,
    RouteType,
    SpecialistType,
)
from ..models.chat import ChatMessage
from ..models.knowledge import KnowledgeRetrieveResponse
from ..models.ollama import ModelGenerationResponse, extract_assistant_text, normalize_generation_response
from ..context.builder import (
    build_controller_messages,
    last_user_text,
    render_request_context,
    render_structured_context,
)
from ..models.manager import ModelManager
from ..schemas import (
    ControllerPlan,
    ControllerValidation,
    CoderResult,
    NormalizedRequest,
    RouteDecision,
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


def _classification_from_hints(hints: RoutingHints) -> str:
    scores = {
        "KNOWLEDGE": hints.repository_likelihood,
        "CODE": hints.code_likelihood,
        "VISION": hints.vision_likelihood,
    }
    classification, score = max(scores.items(), key=lambda item: item[1])
    if score < 0.45:
        return "GENERAL"
    return classification


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


def _infer_next_specialist(
    classification: str,
    *,
    needs_reasoning: bool,
    requires_clarification: bool,
    routing_hints: RoutingHints,
    explicit_next: SpecialistType | None = None,
) -> SpecialistType | None:
    if explicit_next is not None:
        return explicit_next
    if requires_clarification:
        return SpecialistType.CLARIFY
    if classification == "VISION" or routing_hints.vision_likelihood >= 0.45:
        return SpecialistType.VISION
    if classification == "TOOLS":
        return SpecialistType.TOOLS
    if classification == "CODE" or routing_hints.code_likelihood >= 0.55:
        return SpecialistType.CODER
    if classification == "KNOWLEDGE" or routing_hints.repository_likelihood >= 0.55:
        return SpecialistType.KNOWLEDGE
    if needs_reasoning or classification == "REASONING":
        return SpecialistType.REASONING
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
    step = last_step.value if last_step else "unknown"
    summary: dict[str, Any] = {
        "specialist_type": step,
        "execution_status": "unknown",
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
                    "execution_status": "missing_result",
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
                "execution_status": "ok" if hit_count else "no_documents",
                "confidence": knowledge.confidence,
                "result_summary": knowledge.retrieval_reason or (knowledge.context or "")[:400],
                "hit_count": hit_count,
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.CODER:
        coder = _coerce_coder(state.get("coder_result"))
        if coder is None:
            summary.update(
                {
                    "execution_status": "missing_result",
                    "result_summary": "Coder returned no result.",
                }
            )
            return summary
        sufficient = bool((coder.summary or coder.code or "").strip())
        summary.update(
            {
                "execution_status": "ok" if sufficient else "empty_result",
                "confidence": coder.confidence,
                "result_summary": coder.summary or coder.code[:400],
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.TOOLS:
        tool = _coerce_tool(state.get("tool_result"))
        if tool is None:
            summary.update(
                {
                    "execution_status": "missing_result",
                    "result_summary": "Tool step returned no result.",
                }
            )
            return summary
        sufficient = tool.status == "ok" and bool((tool.summary or tool.result or tool.raw_text))
        summary.update(
            {
                "execution_status": tool.status,
                "confidence": 1.0 if sufficient else 0.0,
                "result_summary": tool.summary or str(tool.result)[:400],
                "sufficient": sufficient,
            }
        )
        return summary

    if last_step == SpecialistType.VISION:
        vision_context = str(state.get("vision_context", "") or "").strip()
        summary.update(
            {
                "execution_status": "ok" if vision_context else "empty_result",
                "confidence": 1.0 if vision_context else 0.0,
                "result_summary": vision_context[:400],
                "sufficient": bool(vision_context),
            }
        )
        return summary

    return summary


def plan_to_route(plan: ControllerPlan) -> RouteDecision:
    has_steps = bool(plan.pending_specialists)
    if plan.complete:
        route = RouteType.GENERAL
    elif plan.next_specialist == SpecialistType.CLARIFY:
        route = RouteType.CLARIFY
    elif plan.next_specialist == SpecialistType.VISION:
        route = RouteType.VISION
    elif plan.next_specialist == SpecialistType.CODER:
        route = RouteType.CODE
    elif plan.next_specialist == SpecialistType.KNOWLEDGE:
        route = RouteType.RAG
    elif plan.next_specialist == SpecialistType.TOOLS:
        route = RouteType.TOOLS
    else:
        route = RouteType.MULTI_STEP if has_steps else RouteType.GENERAL

    return RouteDecision(
        route=route,
        confidence=plan.confidence,
        reason=plan.summary or plan.intent or "Controller plan",
        needs_vision=plan.next_specialist == SpecialistType.VISION,
        needs_rag=plan.next_specialist == SpecialistType.KNOWLEDGE,
        needs_tools=plan.next_specialist == SpecialistType.TOOLS,
        needs_code=plan.next_specialist == SpecialistType.CODER,
        needs_planning=has_steps and not plan.complete,
    )


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
        latest_user_message = last_user_text(state.get("messages", []))
        vision_context = state.get("vision_context", "")
        knowledge_result = _coerce_knowledge(state.get("knowledge_result"))
        coder_result = _coerce_coder(state.get("coder_result"))
        tool_result = _coerce_tool(state.get("tool_result"))

        messages = build_controller_messages(
            system_prompt=build_controller_plan_prompt(),
            messages=self._state_messages(state),
            request_context=self._request_context(state),
            vision_context=vision_context,
            knowledge_result=(
                knowledge_result
                if isinstance(knowledge_result, KnowledgeRetrieveResponse)
                else None
            ),
            coder_result=coder_result if isinstance(coder_result, CoderResult) else None,
            tool_result=tool_result if isinstance(tool_result, ToolResult) else None,
            latest_user_message=latest_user_message,
        )

        response = await self.ollama.chat(
            model=self.models.controller().name,
            messages=messages,
            temperature=self.settings.controller_temperature,
            max_tokens=self.settings.controller_final_max_tokens,
            stream=False,
            keep_alive=self.settings.controller_keep_alive,
        )

        parsed = _extract_json_object(response.content)
        hints = _routing_hints_from_state(state)
        hint_classification = _classification_from_hints(hints)
        classification = str(
            parsed.get("classification")
            or parsed.get("route")
            or parsed.get("category")
            or ""
        ).strip().upper()
        known_classifications = {
            "GENERAL",
            "KNOWLEDGE",
            "CODE",
            "VISION",
            "TOOLS",
            "REASONING",
            "CLARIFY",
        }
        if classification not in known_classifications:
            classification = hint_classification
        elif classification == "GENERAL" and hint_classification != "GENERAL":
            classification = hint_classification

        explicit_next = _normalize_controller_step(
            parsed.get("next_specialist") or parsed.get("next_step")
        )
        needs_reasoning = (
            _bool_from_any(parsed.get("needs_reasoning", parsed.get("reasoning_required", False)))
            or classification == "REASONING"
        )
        requires_clarification = classification == "CLARIFY" or _bool_from_any(parsed.get("requires_clarification", False))
        complete = _bool_from_any(parsed.get("complete", False))

        if hint_classification == "GENERAL" and classification == "KNOWLEDGE":
            classification = "GENERAL"
            explicit_next = None
            needs_reasoning = False
            requires_clarification = False
            complete = True

        if classification == "GENERAL":
            explicit_next = None
            needs_reasoning = False
            requires_clarification = False
            complete = True
        elif explicit_next is None:
            explicit_next = _infer_next_specialist(
                classification,
                needs_reasoning=needs_reasoning,
                requires_clarification=requires_clarification,
                routing_hints=hints,
            )
            if explicit_next is None and not needs_reasoning and not requires_clarification:
                complete = True

        pending_specialists = [explicit_next] if explicit_next is not None and not complete else []

        plan = ControllerPlan(
            classification=classification,
            intent=str(parsed.get("intent") or "").strip() or "general",
            summary=str(parsed.get("explanation") or parsed.get("summary") or "").strip() or latest_user_message[:280],
            complexity="medium",
            confidence=_safe_float(parsed.get("confidence"), 0.0),
            action=ControllerAction.FINALIZE if complete else ControllerAction.CONTINUE,
            complete=complete,
            next_specialist=explicit_next,
            pending_specialists=_unique_steps(pending_specialists),
            retry=_bool_from_any(parsed.get("retry", False)),
            retry_reason=str(parsed.get("retry_reason") or parsed.get("reason") or "").strip(),
            needs_reasoning=needs_reasoning,
            final_answer_ready=complete,
            clarification_question=parsed.get("clarification_question"),
            fallback_to_general=classification == "GENERAL" or _bool_from_any(parsed.get("fallback_to_general", False)),
            knowledge_sufficient=None,
            completion_condition=str(parsed.get("completion_condition") or "").strip(),
            explanation=str(parsed.get("explanation") or parsed.get("summary") or "").strip(),
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
        return plan

    async def validate(
        self,
        state: dict[str, Any],
        *,
        last_step: SpecialistType | None = None,
    ) -> ControllerValidation:
        latest_user_message = last_user_text(state.get("messages", []))
        step_text = last_step.value if last_step else "unknown"
        state_summary = self._structured_state_prompt(state)
        evidence = _specialist_evidence_summary(state, last_step=last_step, settings=self.settings)

        validation_messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=build_controller_validation_prompt(),
            ),
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "normalized_request"},
                content=self._request_context(state) or "No normalized request context available.",
            ),
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "state_summary"},
                content=(
                    "Current state summary:\n\n"
                    f"{state_summary or 'No structured context yet.'}"
                ),
            ),
        ]

        if latest_user_message:
            validation_messages.append(
                ChatMessage(
                    role=ChatRole.USER,
                    content=f"Latest user request:\n{latest_user_message}",
                )
            )

        validation_messages.append(
            ChatMessage(
                role=ChatRole.USER,
                content=(
                    f"Last specialist step: {step_text}\n\n"
                    f"Specialist evidence summary:\n{evidence}"
                ),
            )
        )

        response = await self.ollama.chat(
            model=self.models.controller().name,
            messages=validation_messages,
            temperature=0.05,
            max_tokens=self.settings.controller_max_tokens,
            stream=False,
            keep_alive=self.settings.controller_keep_alive,
        )

        parsed = _extract_json_object(response.content)
        hints = _routing_hints_from_state(state)
        parsed_next = _normalize_controller_step(
            parsed.get("next_specialist")
            or parsed.get("next_step")
            or (parsed.get("pending_specialists") or [None])[0]
        )
        plan = _coerce_plan(state.get("execution_plan") or state.get("controller_plan"))
        fallback_next = parsed_next or (plan.next_specialist if plan else None)

        action_raw = str(parsed.get("action") or "continue").strip().lower()
        if action_raw == "reasoning":
            action_raw = "reason"
        try:
            action = ControllerAction(action_raw)
        except Exception:
            action = ControllerAction.CONTINUE

        fallback_to_general = _bool_from_any(parsed.get("fallback_to_general", False))
        knowledge_sufficient = parsed.get("knowledge_sufficient")
        if knowledge_sufficient is not None:
            knowledge_sufficient = _bool_from_any(knowledge_sufficient)
        retry = _bool_from_any(parsed.get("retry", False))
        retry_reason = str(parsed.get("retry_reason") or parsed.get("reason") or "").strip()
        complete = _bool_from_any(parsed.get("complete", False))
        needs_reasoning = action == ControllerAction.REASON or _bool_from_any(
            parsed.get("needs_reasoning", False)
        )
        requires_clarification = action == ControllerAction.CLARIFY
        next_specialist = fallback_next
        if action == ControllerAction.FINALIZE:
            next_specialist = None
            complete = True
        elif action == ControllerAction.CLARIFY:
            next_specialist = SpecialistType.CLARIFY
        elif action == ControllerAction.REASON:
            next_specialist = SpecialistType.REASONING
        elif next_specialist is None and not needs_reasoning:
            complete = True

        if last_step == SpecialistType.KNOWLEDGE:
            sufficient = bool(evidence.get("sufficient", False))
            knowledge_sufficient = sufficient
            request_classification = _classification_from_hints(hints)
            if not sufficient and request_classification == "GENERAL":
                action = ControllerAction.FINALIZE
                fallback_to_general = True
                next_specialist = None
                needs_reasoning = False
                complete = True
            elif (
                not sufficient
                and action in {ControllerAction.FINALIZE, ControllerAction.CONTINUE}
                and next_specialist is None
                and not fallback_to_general
            ):
                if request_classification == "KNOWLEDGE" or hints.repository_likelihood >= 0.55:
                    action = ControllerAction.REASON
                    needs_reasoning = True
                else:
                    action = ControllerAction.CLARIFY
                    needs_reasoning = False
                complete = False
                next_specialist = (
                    SpecialistType.REASONING
                    if action == ControllerAction.REASON
                    else SpecialistType.CLARIFY
                )

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
        retry_counts: dict[str, int] = {}
        for key, value in (state.get("retry_counts", {}) or {}).items():
            try:
                retry_counts[str(key)] = int(value)
            except Exception:
                retry_counts[str(key)] = 0
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

        return ControllerValidation(
            action=action,
            summary=str(parsed.get("summary") or parsed.get("reason") or "").strip(),
            confidence=_safe_float(parsed.get("confidence"), 0.0),
            complete=complete or action == ControllerAction.FINALIZE,
            next_specialist=next_specialist,
            pending_specialists=_unique_steps(pending_specialists),
            retry=retry,
            retry_reason=retry_reason,
            needs_reasoning=needs_reasoning,
            final_answer_ready=complete or action == ControllerAction.FINALIZE,
            fallback_to_general=fallback_to_general,
            knowledge_sufficient=knowledge_sufficient,
            reason=str(parsed.get("reason") or parsed.get("summary") or "").strip(),
            issues=[str(item) for item in (parsed.get("issues") or []) if str(item).strip()],
            notes=str(parsed.get("notes") or "").strip(),
            classification=(plan.classification if plan else "GENERAL"),
        )

    async def finalize(self, state: dict[str, Any]) -> ModelGenerationResponse:
        """
        Produce a final answer with the resident controller.
        If specialist evidence exists, synthesize it into a user-facing answer.
        """
        latest_user_message = last_user_text(state.get("messages", []))
        plan = _coerce_plan(state.get("execution_plan") or state.get("controller_plan"))
        knowledge = _coerce_knowledge(state.get("knowledge_result"))
        coder = _coerce_coder(state.get("coder_result"))
        tool = _coerce_tool(state.get("tool_result"))
        reasoning = _coerce_generation(state.get("reasoning_result"))

        has_specialist_evidence = any(
            [
                knowledge and bool(knowledge.context),
                coder and bool((coder.summary or coder.code).strip()),
                tool and bool((tool.summary or tool.result or tool.raw_text)),
                bool(str(state.get("vision_context", "") or "").strip()),
                reasoning and bool(reasoning.content.strip()),
            ]
        )
        structured_context = ""
        if has_specialist_evidence and not (plan and plan.classification == "GENERAL"):
            structured_context = self._structured_state_prompt(state)

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=build_controller_final_prompt()),
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "structured_context"},
                content=structured_context or "No structured context available.",
            ),
        ]

        if latest_user_message:
            messages.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))

        response = await self.ollama.chat(
            model=self.models.controller().name,
            messages=messages,
            temperature=self.settings.controller_temperature,
            max_tokens=self.settings.controller_max_tokens,
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
