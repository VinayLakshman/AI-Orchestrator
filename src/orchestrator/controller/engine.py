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
from ..models.ollama import ModelGenerationResponse
from ..context.builder import build_controller_messages, last_user_text, render_structured_context
from ..models.manager import ModelManager
from ..schemas import ControllerPlan, ControllerValidation, CoderResult, RouteDecision, ToolResult
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


def _coerce_specialist(value: Any) -> SpecialistType | None:
    if not value:
        return None
    try:
        return SpecialistType(str(value))
    except Exception:
        return None


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


def _text_has_any(text: str, needles: Iterable[str]) -> bool:
    lower = text.lower()
    return any(needle in lower for needle in needles)


def _looks_project_specific(text: str) -> bool:
    return _text_has_any(
        text,
        {
            " my ",
            " our ",
            "this repo",
            "this repository",
            "this project",
            "this codebase",
            "indexed",
            "repository",
            "repositories",
            "local docs",
            "local documentation",
            "docker compose",
            "compose file",
            "my orchestrator",
            "our orchestrator",
            "this orchestrator",
            "knowledge service",
            "metadata reranker",
            "reranker",
            "my implementation",
            "our implementation",
            "this implementation",
            "my config",
            "our config",
            "this config",
            "my configuration",
            "our configuration",
            "this configuration",
            "in the code",
            "in this code",
            "in this repo",
            "in my repo",
        },
    )


def _looks_code_request(text: str) -> bool:
    return _text_has_any(
        text,
        {
            "write code",
            "write a function",
            "implement ",
            "implement a",
            "implement the",
            "debug",
            "fix ",
            "patch",
            "refactor",
            "modify",
            "edit ",
            "review code",
            "generate code",
            "unit test",
            "python code",
            "typescript code",
            "javascript code",
            " in python",
            " in typescript",
            " in javascript",
            "function to",
            "class ",
        },
    )


def _looks_vision_request(text: str) -> bool:
    return _text_has_any(
        text,
        {"image", "screenshot", "photo", "diagram", "picture", "attached", "visual", "ocr"},
    )


def _looks_tools_request(text: str) -> bool:
    return _text_has_any(
        text,
        {"mcp", "tool", "run command", "execute", "call api", "external tool"},
    )


def _looks_reasoning_request(text: str) -> bool:
    return _text_has_any(
        text,
        {
            "architecture",
            "design a plan",
            "migration plan",
            "multi-document",
            "synthesize",
            "tradeoff",
            "trade-off",
            "using my indexed repositories",
            "using indexed repositories",
        },
    )


def _classification_from_text(text: str, *, has_images: bool = False) -> str:
    padded = f" {text.strip().lower()} "
    if has_images or _looks_vision_request(padded):
        return "VISION"
    if _looks_tools_request(padded):
        return "TOOLS"
    if _looks_code_request(padded):
        return "CODE"
    if _looks_project_specific(padded):
        return "KNOWLEDGE"
    if _looks_reasoning_request(padded):
        return "REASONING"
    return "GENERAL"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_complexity(value: Any) -> str:
    complexity = str(value or "medium").strip().lower()
    if complexity in {"low", "medium", "high"}:
        return complexity
    return "medium"


def _normalize_fallback(value: Any) -> str:
    fallback = str(value or "general").strip().lower()
    if fallback in {"general", "reasoning", "clarify", "none"}:
        return fallback
    return "general"


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

    if plan and plan.requires_clarification:
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
    has_steps = bool(plan.execution_steps)
    if plan.requires_clarification:
        route = RouteType.CLARIFY
    elif plan.requires_vision:
        route = RouteType.VISION
    elif plan.requires_coder:
        route = RouteType.CODE
    elif plan.requires_knowledge:
        route = RouteType.RAG
    elif plan.requires_tools:
        route = RouteType.TOOLS
    elif has_steps:
        route = RouteType.MULTI_STEP
    else:
        route = RouteType.GENERAL

    return RouteDecision(
        route=route,
        confidence=plan.confidence,
        reason=plan.summary or plan.intent or "Controller plan",
        needs_vision=plan.requires_vision,
        needs_rag=plan.requires_knowledge,
        needs_tools=plan.requires_tools,
        needs_code=plan.requires_coder,
        needs_planning=has_steps,
    )


@dataclass(slots=True)
class ControllerEngine:
    settings: Settings
    ollama: OllamaClient
    models: ModelManager

    def _state_messages(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return list(state.get("messages", []) or [])

    def _structured_state_prompt(self, state: dict[str, Any]) -> str:
        plan = _coerce_plan(state.get("controller_plan"))
        validation = _coerce_validation(state.get("controller_validation"))
        knowledge_result = state.get("knowledge_result")
        vision_context = state.get("vision_context", "")
        coder_result = state.get("coder_result")
        tool_result = state.get("tool_result")
        reasoning_result = state.get("reasoning_result")
        return render_structured_context(
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
            max_tokens=self.settings.controller_max_tokens,
            stream=False,
            keep_alive=self.settings.controller_keep_alive,
        )

        parsed = _extract_json_object(response.content)
        raw_steps = parsed.get("execution_steps", parsed.get("steps", []))
        execution_steps = [
            step
            for step in (_coerce_specialist(item) for item in raw_steps)
            if step is not None
        ]
        execution_steps = _unique_steps(execution_steps)
        classification = str(
            parsed.get("classification")
            or parsed.get("route")
            or parsed.get("category")
            or ""
        ).strip().upper()
        heuristic_classification = _classification_from_text(
            latest_user_message,
            has_images=bool(state.get("has_images", False)),
        )
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
            classification = heuristic_classification

        # Keep common public explanations out of the retrieval path even when
        # the small controller over-eagerly asks for knowledge.
        if heuristic_classification == "GENERAL" and classification == "KNOWLEDGE":
            classification = "GENERAL"
            execution_steps = [step for step in execution_steps if step != SpecialistType.KNOWLEDGE]

        if classification == "GENERAL":
            execution_steps = []
        elif classification == "KNOWLEDGE" and SpecialistType.KNOWLEDGE not in execution_steps:
            execution_steps.append(SpecialistType.KNOWLEDGE)
        elif classification == "CODE" and SpecialistType.CODER not in execution_steps:
            execution_steps.append(SpecialistType.CODER)
        elif classification == "VISION" and SpecialistType.VISION not in execution_steps:
            execution_steps.append(SpecialistType.VISION)
        elif classification == "TOOLS" and SpecialistType.TOOLS not in execution_steps:
            execution_steps.append(SpecialistType.TOOLS)

        plan = ControllerPlan(
            classification=classification,
            intent=str(parsed.get("intent") or "").strip() or "general",
            summary=(
                str(parsed.get("summary") or parsed.get("explanation") or "").strip()
                or latest_user_message[:280]
            ),
            complexity=_normalize_complexity(parsed.get("complexity")),
            confidence=_safe_float(parsed.get("confidence"), 0.0),
            requires_vision=SpecialistType.VISION in execution_steps or classification == "VISION",
            requires_knowledge=(
                SpecialistType.KNOWLEDGE in execution_steps or classification == "KNOWLEDGE"
            ),
            requires_coder=SpecialistType.CODER in execution_steps or classification == "CODE",
            requires_tools=SpecialistType.TOOLS in execution_steps or classification == "TOOLS",
            requires_reasoning=(
                classification == "REASONING"
                or _bool_from_any(
                    parsed.get("requires_reasoning", parsed.get("reasoning_required", False))
                )
                or (
                    classification == "KNOWLEDGE"
                    and _looks_reasoning_request(f" {latest_user_message.lower()} ")
                )
            ),
            requires_clarification=(
                classification == "CLARIFY"
                or _bool_from_any(parsed.get("requires_clarification", False))
            ),
            clarification_question=parsed.get("clarification_question"),
            tool_requests=list(parsed.get("tool_requests") or []),
            execution_steps=_unique_steps(execution_steps),
            fallback=_normalize_fallback(parsed.get("fallback")),
            completion_condition=str(parsed.get("completion_condition") or "").strip(),
            explanation=str(parsed.get("explanation") or parsed.get("summary") or "").strip(),
        )

        if not plan.execution_steps:
            fallback_steps: list[SpecialistType] = []
            if plan.requires_knowledge and classification != "GENERAL":
                fallback_steps.append(SpecialistType.KNOWLEDGE)
            if plan.requires_vision:
                fallback_steps.append(SpecialistType.VISION)
            if plan.requires_coder:
                fallback_steps.append(SpecialistType.CODER)
            if plan.requires_tools:
                fallback_steps.append(SpecialistType.TOOLS)
            plan.execution_steps = _unique_steps(fallback_steps)

        if plan.classification == "GENERAL":
            plan.requires_knowledge = False
            plan.requires_vision = False
            plan.requires_coder = False
            plan.requires_tools = False
            plan.requires_reasoning = False
            plan.requires_clarification = False
            plan.execution_steps = []

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
        next_steps = [
            step
            for step in (_coerce_specialist(item) for item in parsed.get("next_steps", []))
            if step is not None
        ]

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

        if last_step == SpecialistType.KNOWLEDGE:
            sufficient = bool(evidence.get("sufficient", False))
            knowledge_sufficient = sufficient
            request_classification = _classification_from_text(
                latest_user_message,
                has_images=bool(state.get("has_images", False)),
            )
            if not sufficient and request_classification == "GENERAL":
                action = ControllerAction.FINALIZE
                fallback_to_general = True
                next_steps = []
            elif (
                not sufficient
                and action in {ControllerAction.FINALIZE, ControllerAction.CONTINUE}
                and not next_steps
                and not fallback_to_general
            ):
                if _looks_reasoning_request(f" {latest_user_message.lower()} "):
                    action = ControllerAction.REASON
                else:
                    action = ControllerAction.CLARIFY
                next_steps = []

        return ControllerValidation(
            action=action,
            summary=str(parsed.get("summary") or parsed.get("reason") or "").strip(),
            confidence=_safe_float(parsed.get("confidence"), 0.0),
            needs_reasoning=(
                action == ControllerAction.REASON
                or _bool_from_any(parsed.get("needs_reasoning", False))
            ),
            final_answer_ready=(
                action == ControllerAction.FINALIZE
                or _bool_from_any(parsed.get("final_answer_ready", False))
            ),
            next_steps=_unique_steps(next_steps),
            fallback_to_general=fallback_to_general,
            knowledge_sufficient=knowledge_sufficient,
            reason=str(parsed.get("reason") or parsed.get("summary") or "").strip(),
            issues=[str(item) for item in (parsed.get("issues") or []) if str(item).strip()],
            notes=str(parsed.get("notes") or "").strip(),
        )

    async def finalize(self, state: dict[str, Any]) -> ModelGenerationResponse:
        """
        Produce a final answer with the resident controller.
        If specialist evidence exists, synthesize it into a user-facing answer.
        """
        latest_user_message = last_user_text(state.get("messages", []))
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
        if not response.content.strip():
            return ModelGenerationResponse(
                model=response.model,
                content=_fallback_final_answer(state),
                raw=response.raw,
            )
        return response

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
        )
        return response
