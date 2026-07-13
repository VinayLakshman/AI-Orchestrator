from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..clients.ollama import OllamaClient
from ..context.builder import build_controller_messages, last_user_text, render_structured_context
from ..models.manager import ModelManager
from ..schemas import (
    ControllerAction,
    ControllerPlan,
    ControllerValidation,
    CoderResult,
    KnowledgeRetrieveResponse,
    ModelGenerationResponse,
    RouteDecision,
    RouteType,
    SpecialistType,
    ToolResult,
)
from ..models.chat import ChatRole, ChatMessage
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
        return render_structured_context(
            vision_context=vision_context,
            knowledge_result=_coerce_knowledge(knowledge_result),
            coder_result=_coerce_coder(coder_result),
            tool_result=_coerce_tool(tool_result),
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
            knowledge_result=knowledge_result if isinstance(knowledge_result, KnowledgeRetrieveResponse) else None,
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
        execution_steps = [
            step
            for step in (_coerce_specialist(item) for item in parsed.get("execution_steps", []))
            if step is not None
        ]
        execution_steps = _unique_steps(execution_steps)

        plan = ControllerPlan(
            intent=str(parsed.get("intent") or "").strip() or "general",
            summary=str(parsed.get("summary") or "").strip() or latest_user_message[:280],
            complexity=str(parsed.get("complexity") or "medium"),
            confidence=float(parsed.get("confidence") or 0.0),
            requires_vision=bool(parsed.get("requires_vision", False)),
            requires_knowledge=bool(parsed.get("requires_knowledge", False)),
            requires_coder=bool(parsed.get("requires_coder", False)),
            requires_tools=bool(parsed.get("requires_tools", False)),
            requires_reasoning=bool(parsed.get("requires_reasoning", False)),
            requires_clarification=bool(parsed.get("requires_clarification", False)),
            clarification_question=parsed.get("clarification_question"),
            tool_requests=list(parsed.get("tool_requests") or []),
            execution_steps=execution_steps,
        )

        if not plan.execution_steps:
            fallback_steps: list[SpecialistType] = []
            if plan.requires_knowledge:
                fallback_steps.append(SpecialistType.KNOWLEDGE)
            if plan.requires_vision:
                fallback_steps.append(SpecialistType.VISION)
            if plan.requires_coder:
                fallback_steps.append(SpecialistType.CODER)
            if plan.requires_tools:
                fallback_steps.append(SpecialistType.TOOLS)
            plan.execution_steps = _unique_steps(fallback_steps)

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

        validation_messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=build_controller_validation_prompt(),
            ),
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "state_summary"},
                content=f"Current state summary:\n\n{state_summary or 'No structured context yet.'}",
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
                content=f"Last specialist step: {step_text}",
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

        try:
            action = ControllerAction(str(parsed.get("action") or "continue"))
        except Exception:
            action = ControllerAction.CONTINUE

        return ControllerValidation(
            action=action,
            summary=str(parsed.get("summary") or "").strip(),
            confidence=float(parsed.get("confidence") or 0.0),
            needs_reasoning=bool(parsed.get("needs_reasoning", False)),
            final_answer_ready=bool(parsed.get("final_answer_ready", False)),
            next_steps=_unique_steps(next_steps),
            issues=[str(item) for item in (parsed.get("issues") or []) if str(item).strip()],
            notes=str(parsed.get("notes") or "").strip(),
        )

    async def finalize(self, state: dict[str, Any]) -> ModelGenerationResponse:
        """
        Produce a final answer with the resident controller.
        If a reasoning result already exists, use that content directly.
        """
        reasoning_result = state.get("reasoning_result")
        if isinstance(reasoning_result, dict):
            try:
                reasoning_result = ModelGenerationResponse.model_validate(reasoning_result)
            except Exception:
                reasoning_result = None
        if isinstance(reasoning_result, ModelGenerationResponse) and reasoning_result.content.strip():
            return reasoning_result

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