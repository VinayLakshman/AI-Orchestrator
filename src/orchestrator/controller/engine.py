from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from orchestrator.models.state import OrchestratorState
from orchestrator.streaming.publisher import StreamPublisher

from ..clients.ollama import OllamaClient
from ..common.enums import ChatRole, SpecialistType
from ..common.utils import _extract_json_object
from ..context.builder import (
    build_controller_messages,
    build_finalize_context,
    build_finalizer_messages,
    estimate_text_tokens,
    render_request_context,
    render_structured_context,
)
from ..logging import get_logger
from ..models.chat import ChatMessage
from ..models.manager import ModelManager
from ..models.ollama import (
    ModelGenerationResponse,
    extract_assistant_text,
    normalize_generation_response,
)
from ..models.execution import ExecutionPlan, ValidationResult
from ..settings import Settings
from .prompts import (
    build_controller_final_prompt,
    build_controller_plan_prompt,
    build_controller_validation_prompt,
    build_reasoning_prompt,
)

logger = get_logger(__name__)

_ALLOWED_CLASSIFICATIONS = {
    "GENERAL",
    "KNOWLEDGE",
    "CODE",
    "VISION",
    "TOOLS",
    "REASONING",
    "CLARIFY",
}

_ALLOWED_ACTIONS = {
    "continue",
    "finalize",
    "reason",
    "clarify",
}

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


def _unique_steps(steps: Iterable[Any]) -> list[SpecialistType]:
    seen: set[str] = set()
    out: list[SpecialistType] = []
    for step in steps:
        normalized = _normalize_step(step)
        if normalized is None:
            continue
        key = normalized.value
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }

    return bool(value)


def _request_messages(
        state: OrchestratorState,
    ) -> list[ChatMessage]:
    return list(state.request.messages)


def _normalized_plan_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    classification = str(
        parsed.get("classification")
        or parsed.get("route")
        or parsed.get("category")
        or "GENERAL"
    ).strip().upper()

    if classification == "RAG":
        classification = "KNOWLEDGE"
    elif classification == "REASON":
        classification = "REASONING"

    if classification not in _ALLOWED_CLASSIFICATIONS:
        classification = "GENERAL"

    raw_queue = (
        parsed.get("execution_queue")
        or parsed.get("pending_specialists")
        or []
    )
    if isinstance(raw_queue, str):
        raw_queue = [raw_queue]
    if not raw_queue and parsed.get("next_specialist"):
        raw_queue = [parsed.get("next_specialist")]

    queue = _unique_steps(raw_queue)

    return {
        "classification": classification,
        "intent": str(parsed.get("intent") or "").strip(),
        "summary": str(parsed.get("summary") or "").strip(),
        "complexity": str(parsed.get("complexity") or "medium").strip().lower(),
        "confidence": _safe_float(parsed.get("confidence"), 0.0),
        "action": str(parsed.get("action") or "continue").strip().lower(),
        "complete": _bool_from_any(parsed.get("complete")),
        "next_specialist": _normalize_step(
            parsed.get("next_specialist") or parsed.get("next_step")
        ),
        "pending_specialists": queue,
        "execution_queue": queue,
        "requires_repository": SpecialistType.KNOWLEDGE in queue,
        "requires_web": SpecialistType.WEB in queue,
        "requires_vision": SpecialistType.VISION in queue,
        "requires_code": SpecialistType.CODER in queue,
        "requires_tools": SpecialistType.TOOLS in queue,
        "requires_reasoning": SpecialistType.REASONING in queue,
        "retry": _bool_from_any(parsed.get("retry")),
        "retry_reason": str(parsed.get("retry_reason") or "").strip(),
        "needs_reasoning": _bool_from_any(
            parsed.get("needs_reasoning") or parsed.get("requires_reasoning")
        ),
        "final_answer_ready": _bool_from_any(parsed.get("final_answer_ready")),
        "clarification_question": (
            str(parsed.get("clarification_question") or "").strip() or None
        ),
        "fallback_to_general": _bool_from_any(parsed.get("fallback_to_general")),
        "knowledge_sufficient": parsed.get("knowledge_sufficient"),
        "use_web_search": _bool_from_any(
            parsed.get("use_web_search") or parsed.get("requires_web")
        ),
        "tool_requests": parsed.get("tool_requests") or [],
        "completion_condition": str(parsed.get("completion_condition") or "").strip(),
        "explanation": str(parsed.get("explanation") or "").strip(),
        "reason": str(parsed.get("reason") or "").strip(),
        "issues": list(parsed.get("issues") or []),
        "notes": str(parsed.get("notes") or "").strip(),
        "route": (
            "rag"
            if str(parsed.get("route") or classification).strip().upper() in {"KNOWLEDGE", "RAG"}
            else str(parsed.get("route") or classification).strip().lower()
        ),
        "route_hint": parsed.get("route_hint"),
    }


def _normalized_validation_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    action = str(parsed.get("action") or "continue").strip().lower()
    if action == "reasoning":
        action = "reason"
    if action not in _ALLOWED_ACTIONS:
        action = "continue"

    return {
        "action": action,
        "confidence": _safe_float(parsed.get("confidence"), 0.0),
        "complete": _bool_from_any(parsed.get("complete")),
        "summary": str(parsed.get("summary") or "").strip(),
        "retry": _bool_from_any(parsed.get("retry")),
        "retry_reason": str(parsed.get("retry_reason") or "").strip(),
        "requires_reasoning": _bool_from_any(
            parsed.get("needs_reasoning")
            or parsed.get("requires_reasoning")
        ),
        "requires_clarification": _bool_from_any(
            parsed.get("requires_clarification")
        ),
        "fallback_to_general": _bool_from_any(
            parsed.get("fallback_to_general")
        ),
        "reason": str(parsed.get("reason") or "").strip(),
        "issues": list(parsed.get("issues") or []),
        "notes": str(parsed.get("notes") or "").strip(),
        "next_specialist": _normalize_step(
            parsed.get("next_specialist") or parsed.get("next_step")
        ),
        "pending_specialists": _unique_steps(
            parsed.get("pending_specialists") or []
        ),
        "execution_queue": _unique_steps(
            parsed.get("execution_queue")
            or parsed.get("pending_specialists")
            or []
        ),
        "final_answer_ready": _bool_from_any(parsed.get("final_answer_ready")),
        "knowledge_sufficient": parsed.get("knowledge_sufficient"),
        "use_web_search": _bool_from_any(
            parsed.get("use_web_search") or parsed.get("requires_web")
        ),
        "classification": str(parsed.get("classification") or "GENERAL").strip().upper(),
    }


@dataclass(slots=True)
class ControllerEngine:
    settings: Settings
    ollama: OllamaClient
    models: ModelManager

    async def plan(self, state: OrchestratorState) -> ExecutionPlan:
        profile = {
            "has_images": bool(state.request.images),
            "has_files": bool(state.request.metadata.get("attachments")),
        }

        messages = build_controller_messages(
            system_prompt=build_controller_plan_prompt(),
            messages=_request_messages(state),
            request_context=render_request_context(state.request),
        )

        response = await self.ollama.chat(
            model=self.models.controller().name,
            messages=messages,
            temperature=self.settings.controller_plan_temperature,
            max_tokens=self.settings.controller_plan_max_tokens,
            stream=False,
            keep_alive=self.settings.controller_keep_alive,
        )

        raw_content = response.content or response.raw or "{}"
        parsed = _extract_json_object(raw_content)
        if not isinstance(parsed, dict):
            parsed = {}

        payload = _normalized_plan_payload(parsed)

        try:
            plan = ExecutionPlan.model_validate(payload)
        except Exception:
            logger.exception("failed_to_validate_execution_plan")
            plan = ExecutionPlan()

        if plan.execution_queue:
            plan.execution_queue = _unique_steps(plan.execution_queue)

        logger.debug(
            "execution_plan %s",
            json.dumps(
                {
                    "classification": plan.classification,
                    "queue": [s.value for s in plan.execution_queue],
                    "repository": plan.requires_repository,
                    "web": plan.requires_web,
                    "vision": plan.requires_vision,
                    "tools": plan.requires_tools,
                    "code": plan.requires_code,
                    "reasoning": plan.requires_reasoning,
                    "request_evidence": profile,
                },
                sort_keys=True,
                default=str,
            ),
        )

        return plan

    async def validate(
        self,
        state: OrchestratorState,
        *,
        last_step: SpecialistType | None = None,
    ) -> ValidationResult:
        execution = state.execution
        runtime = execution.runtime

        step_text = last_step.value if last_step else "unknown"

        validation_messages = build_controller_messages(
            system_prompt=build_controller_validation_prompt(),
            messages=_request_messages(state),
            request_context=render_request_context(state.request),
            structured_context=render_structured_context(state),
        )

        response = await self.ollama.chat(
            model=self.models.controller().name,
            messages=validation_messages,
            temperature=self.settings.controller_validate_temperature,
            max_tokens=self.settings.controller_validate_max_tokens,
            stream=False,
            keep_alive=self.settings.controller_keep_alive,
        )

        raw_content = response.content or response.raw or "{}"
        parsed = _extract_json_object(raw_content)
        if not isinstance(parsed, dict):
            parsed = {}

        payload = _normalized_validation_payload(parsed)

        try:
            validation = ValidationResult.model_validate(payload)
        except Exception:
            logger.exception("failed_to_validate_execution_result")
            validation = ValidationResult()

        logger.debug(
            "execution_validation %s",
            json.dumps(
                {
                    "last_step": step_text,
                    "action": validation.action.value,
                    "complete": validation.complete,
                    "retry": validation.retry,
                    "current_index": runtime.current_index,
                    "completed": [
                        item.value
                        for item in runtime.completed
                    ],
                },
                sort_keys=True,
                default=str,
            ),
        )

        return validation

    async def finalize(
        self,
        state: OrchestratorState,
        publisher: StreamPublisher | None = None,
    ) -> ModelGenerationResponse:
        finalizer_context = build_finalize_context(state)

        context_json = json.dumps(
            finalizer_context,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        finalizer_prompt = build_controller_final_prompt()

        messages = build_finalizer_messages(
            system_prompt=finalizer_prompt,
            messages=_request_messages(state),
            evidence_context=context_json,
        )

        logger.debug(
            "finalizer_context %s",
            json.dumps(
                {
                    "prompt_tokens": estimate_text_tokens(finalizer_prompt),
                    "context_tokens": estimate_text_tokens(context_json),
                    "sources": len(finalizer_context.get("sources", [])),
                },
                sort_keys=True,
                default=str,
            ),
        )

        model_name = self.models.controller().name

        try:
            if publisher is not None:
                await publisher.llm_started(model=model_name)

                parts: list[str] = []
                raw: dict[str, Any] | None = None

                async for chunk in self.ollama.stream_chat(
                    model=model_name,
                    messages=messages,
                    temperature=self.settings.controller_finalize_temperature,
                    max_tokens=self.settings.controller_finalize_max_tokens,
                    keep_alive=self.settings.controller_keep_alive,
                ):
                    if chunk.content:
                        parts.append(chunk.content)
                        await publisher.llm_token(chunk.content)

                    if chunk.raw:
                        raw = chunk.raw

                await publisher.llm_finished()

                return ModelGenerationResponse(
                    model=model_name,
                    content=extract_assistant_text("".join(parts)).strip(),
                    raw=raw,
                )

            response = await self.ollama.chat(
                model=model_name,
                messages=messages,
                temperature=self.settings.controller_finalize_temperature,
                max_tokens=self.settings.controller_finalize_max_tokens,
                stream=False,
                keep_alive=self.settings.controller_keep_alive,
            )

            return ModelGenerationResponse(
                model=response.model,
                content=extract_assistant_text(response.content).strip(),
                raw=response.raw,
            )

        except Exception as exc:
            if publisher is not None:
                await publisher.error(str(exc), stage="finalize")
            raise

    async def reason(
        self,
        state: OrchestratorState,
    ) -> ModelGenerationResponse:
        structured_context = render_structured_context(state)
        latest_user_message = state.request.user_message

        messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=build_reasoning_prompt(),
            ),
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "orchestrator_state"},
                content=structured_context,
            ),
        ]

        if latest_user_message:
            messages.append(
                ChatMessage(
                    role=ChatRole.USER,
                    content=latest_user_message,
                )
            )

        response = await self.ollama.chat(
            model=self.models.reasoning().name,
            messages=messages,
            temperature=self.settings.reasoning_temperature,
            max_tokens=self.settings.reasoning_max_tokens,
            stream=False,
            keep_alive=self.settings.reasoning_keep_alive,
            think=self.settings.reasoning_think,
        )

        return normalize_generation_response(
            response.model,
            response.content,
        )
