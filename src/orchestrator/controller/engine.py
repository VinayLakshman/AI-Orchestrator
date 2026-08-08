from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from orchestrator.models.state import OrchestratorState
from orchestrator.streaming.publisher import StreamPublisher

from ..common.enums import ChatRole, SpecialistType
from ..common.utils import _extract_json_object
from ..context.assembler import build_conversation
from ..context.builder import (
    build_controller_messages,
    build_finalize_context,
    build_finalizer_messages,
    estimate_text_tokens,
    render_request_context,
    render_structured_context,
)
from ..context.parser import split_conversation
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
    """Normalize a planner specialist token into SpecialistType.

    Returns None for empty values only.
    Unknown specialist tokens are handled by _unique_steps where we can
    enforce strict contract behavior.
    """
    if value is None:
        return None
    if isinstance(value, SpecialistType):
        return value
    text = str(value).strip()
    if not text:
        return None

    # SpecialistType is a StrEnum; this will raise ValueError if unknown.
    return SpecialistType(text.lower())



def _unique_steps(steps: Iterable[Any]) -> list[SpecialistType]:
    """Validate and de-duplicate planner execution queue entries.

    Contract: every planner-supplied entry must map to a supported
    SpecialistType. Unknown entries are treated as a contract violation
    and will raise.

    Duplicate valid specialists are de-duplicated (preserves existing behavior).
    """

    seen: set[str] = set()
    out: list[SpecialistType] = []

    raw_steps = list(steps)
    invalid: list[Any] = []

    for step in raw_steps:
        try:
            normalized = _normalize_step(step)
        except ValueError:
            invalid.append(step)
            continue

        if normalized is None:
            continue

        key = normalized.value
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)

    if invalid:
        logger.error(
            "planner_contract_violation invalid_specialist_detected invalid_tokens=%s raw_execution_queue=%s",
            invalid,
            raw_steps,
        )
        raise ValueError(f"Invalid execution_queue specialist token(s): {invalid}")

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


def _response_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    raw = getattr(response, "raw", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, dict) and raw:
        return json.dumps(raw, sort_keys=True, default=str)

    return ""


def _log_planner_request(
    *,
    system_prompt: str,
    request_context: str,
    messages: list[ChatMessage],
) -> None:
    latest_user_prompt = ""
    for message in reversed(messages):
        if message.role == ChatRole.USER:
            latest_user_prompt = str(message.content or "").strip()
            break

    logger.debug(
        "\n=========================\nPlanner Request\n=========================\n\nSystem Prompt\n%s\n\nRequest Context\n%s\n\nUser Prompt\n%s\n",
        system_prompt,
        request_context,
        latest_user_prompt,
    )


def _log_planner_response(raw_response: str) -> None:
    logger.debug(
        "\n=========================\nRaw LLM Response\n=========================\n\n%s\n",
        raw_response or "<empty>",
    )


def _log_planner_plan(plan: ExecutionPlan) -> None:
    logger.debug(
        "\n=========================\nParsed ExecutionPlan\n=========================\n\n%s\n",
        json.dumps(plan.model_dump(exclude_none=True), indent=2, sort_keys=True, default=str),
    )


def _normalize_route_value(parsed: dict[str, Any], *, classification: str) -> str:

    """Normalize planner `route` into a valid coarse RouteType.

    Reasoning must NOT be allowed to leak into the graph route.
    If planner emits an invalid route, fail open only to a safe coarse route
    while keeping specialist intent via `requires_reasoning` / `execution_queue`.
    """

    from ..common.enums import RouteType

    raw = parsed.get("route") or parsed.get("graph_route") or parsed.get("category")
    candidate = str(raw or "").strip().lower() or str(classification).strip().lower()

    specialist_tokens = {
        "reasoning",
        "clarify",
        "knowledge",
        "web",
        "vision",
        "code",
        "coder",
        "tools",
        "tool",
        "rag",
        "multi_step",
        "multistep",
    }

    if candidate in specialist_tokens:
        return RouteType.MULTI_STEP.value if candidate == "multi_step" else RouteType.GENERAL.value

    for rt in RouteType:
        if candidate == rt.value:
            return rt.value

    # default safe coarse route
    return RouteType.GENERAL.value


def _normalized_plan_payload(parsed: dict[str, Any]) -> dict[str, Any]:

    # Preserve execution queue exactly as produced by the planner (except for
    # safe normalization like casing and de-duplication).
    # Important: do not alter requires_* booleans based on `route`.

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

    # If planner set explicit requires_* booleans but omitted execution_queue,
    # infer the execution queue from those flags (without keyword heuristics).
    if not queue:
        inferred_steps: list[SpecialistType] = []
        if _bool_from_any(parsed.get("requires_repository") or parsed.get("requires_rag")):
            inferred_steps.append(SpecialistType.KNOWLEDGE)
        if _bool_from_any(parsed.get("requires_web")):
            inferred_steps.append(SpecialistType.WEB)
        if _bool_from_any(parsed.get("requires_vision")):
            inferred_steps.append(SpecialistType.VISION)
        # planner prompt may use requires_code to mean coder
        if _bool_from_any(parsed.get("requires_code")):
            inferred_steps.append(SpecialistType.CODER)
        if _bool_from_any(parsed.get("requires_tools")):
            inferred_steps.append(SpecialistType.TOOLS)
        if _bool_from_any(parsed.get("requires_reasoning")):
            inferred_steps.append(SpecialistType.REASONING)

        queue = _unique_steps(inferred_steps)

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
        # NOTE: `route` is a coarse orchestration route only.
        # Do not allow specialist values (e.g. `reasoning`) to leak into the graph route.
        # If planner emits an invalid route (including specialist-ish tokens), we default
        # to a safe coarse route while preserving requires_reasoning/execution_queue.
        "route": _normalize_route_value(parsed, classification=classification),

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
    models: ModelManager
    lifecycle: Any = None

    async def _ensure_controller_warm(self) -> None:
        """Ensure the controller container owns the GPU before any controller call.

        With transient (single-GPU-owner) containers, the controller may have been
        unloaded in favor of another model. Re-warm it here so plan/validate/finalize
        always have a resident controller.
        """
        if self.lifecycle is not None:
            await self.lifecycle.ensure_warm("controller")

    async def plan(self, state: OrchestratorState) -> ExecutionPlan:
        await self._ensure_controller_warm()
        profile = {
            "has_images": bool(state.request.images),
            "has_files": bool(state.request.metadata.get("attachments")),
        }

        system_prompt = build_controller_plan_prompt()
        request_context = render_request_context(state.request)

        messages = build_controller_messages(
            system_prompt=system_prompt,
            messages=_request_messages(state),
            request_context=request_context,
        )

        logger.debug(
            "planner_model_request model=%s temperature=%s max_tokens=%s json_mode=%s",
            self.models.controller().name,
            self.settings.controller_plan_temperature,
            self.settings.controller_plan_max_tokens,
            True,
        )

        _log_planner_request(
            system_prompt=system_prompt,
            request_context=request_context,
            messages=messages,
        )

        response = await self.models.client("controller").chat(
            model=self.models.controller().name,
            messages=messages,
            temperature=self.settings.controller_plan_temperature,
            max_tokens=self.settings.controller_plan_max_tokens,
            stream=False,
            response_format="json",
            keep_alive=self.settings.controller_keep_alive,
        )

        raw_content = _response_text(response)
        _log_planner_response(raw_content)
        parsed = _extract_json_object(raw_content)
        if not isinstance(parsed, dict):
            parsed = {}

        if not parsed:
            logger.error(
                "planner_response_parse_failed model=%s raw_response=%s",
                self.models.controller().name,
                raw_content[:4000] or "<empty>",
            )

        payload = _normalized_plan_payload(parsed)

        try:
            plan = ExecutionPlan.model_validate(payload)
        except Exception:
            logger.exception(
                "failed_to_validate_execution_plan raw_response=%s",
                raw_content[:4000],
            )
            plan = ExecutionPlan()

        if plan.execution_queue:
            try:
                plan.execution_queue = _unique_steps(plan.execution_queue)
            except ValueError:
                # planner emitted an invalid execution queue; fail loudly
                # so the system can recover via the existing exception path.
                logger.exception(
                    "planner_contract_validation_failed invalid_execution_queue raw_response=%s",
                    raw_content[:4000] or "<empty>",
                )
                raise


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

        _log_planner_plan(plan)

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

        await self._ensure_controller_warm()

        validation_messages = build_controller_messages(
            system_prompt=build_controller_validation_prompt(),
            messages=_request_messages(state),
            request_context=render_request_context(state.request),
            structured_context=render_structured_context(state),
        )

        response = await self.models.client("controller").chat(
            model=self.models.controller().name,
            messages=validation_messages,
            temperature=self.settings.controller_validate_temperature,
            max_tokens=self.settings.controller_validate_max_tokens,
            stream=False,
            response_format="json",
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
        await self._ensure_controller_warm()

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
        controller_client = self.models.client("controller")

        try:
            if publisher is not None:
                await publisher.llm_started(model=model_name)

                parts: list[str] = []
                raw: dict[str, Any] | None = None

                async for chunk in controller_client.stream_chat(
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

            response = await controller_client.chat(
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

        # Build conversation history through the single authoritative
        # ConversationContextBuilder (via split_conversation, which uses the
        # token-budget-driven builder behind the scenes). This keeps the
        # Reasoning Specialist consistent with every other orchestrator node.
        history_messages: list[ChatMessage] = []
        try:
            history_messages, _, _ = split_conversation(state.request.messages)
        except ValueError:
            history_messages = []

        # Delegate to the centralized assembler. Exactly one SYSTEM message is
        # emitted, the structured context is merged into it, and the latest
        # user request becomes a real USER message (compatible with llama.cpp,
        # Qwen and other OpenAI-compatible chat APIs). History is passed
        # through unchanged and chronologically ordered.
        messages = build_conversation(
            system_prompt=build_reasoning_prompt(),
            structured_context=structured_context,
            history=history_messages,
            latest_user_message=latest_user_message,
        )

        response = await self.models.client("reasoning").chat(
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
