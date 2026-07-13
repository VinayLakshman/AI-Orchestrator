from __future__ import annotations

import json
from typing import Any

from ..clients.knowledge import KnowledgeClient
from ..clients.ollama import OllamaClient
from ..common.enums import ChatRole, ControllerAction, SpecialistType
from ..context.builder import last_user_text, render_structured_context
from ..controller.engine import ControllerEngine, plan_to_route
from ..models.chat import ChatMessage
from ..models.knowledge import KnowledgeRetrieveResponse
from ..models.ollama import ModelGenerationResponse
from ..schemas import ControllerPlan, ControllerValidation, CoderResult, ToolResult
from ..logging import get_logger
from ..settings import Settings
from ..streaming.context import get_current_stream
from ..vision.fetcher import collect_latest_message_images, strip_images_from_messages
from ..vision.pipeline import VisionPipeline
from ..graph.state import OrchestratorState


logger = get_logger(__name__)


def _as_plan(value: Any) -> ControllerPlan | None:
    if value is None:
        return None
    if isinstance(value, ControllerPlan):
        return value
    if isinstance(value, dict):
        return ControllerPlan.model_validate(value)
    return None


def _as_validation(value: Any) -> ControllerValidation | None:
    if value is None:
        return None
    if isinstance(value, ControllerValidation):
        return value
    if isinstance(value, dict):
        return ControllerValidation.model_validate(value)
    return None


def _as_knowledge(value: Any) -> KnowledgeRetrieveResponse | None:
    if value is None:
        return None
    if isinstance(value, KnowledgeRetrieveResponse):
        return value
    if isinstance(value, dict):
        return KnowledgeRetrieveResponse.model_validate(value)
    return None


def _as_coder(value: Any) -> CoderResult | None:
    if value is None:
        return None
    if isinstance(value, CoderResult):
        return value
    if isinstance(value, dict):
        return CoderResult.model_validate(value)
    return None


def _as_tool(value: Any) -> ToolResult | None:
    if value is None:
        return None
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, dict):
        return ToolResult.model_validate(value)
    return None


def _step_from_pending(pending_steps: list[str] | None) -> SpecialistType | None:
    if not pending_steps:
        return None
    try:
        return SpecialistType(pending_steps[0])
    except Exception:
        return None


def _state_snapshot(state: OrchestratorState) -> dict[str, Any]:
    return {
        "current_step": state.get("current_step", ""),
        "pending_steps": list(state.get("pending_steps", []) or []),
        "completed_steps": list(state.get("completed_steps", []) or []),
        "controller_cycles": int(state.get("controller_cycles", 0) or 0),
        "specialist_executions": int(state.get("specialist_executions", 0) or 0),
        "workflow_stall_count": int(state.get("workflow_stall_count", 0) or 0),
    }


def _log_transition(event: str, **payload: Any) -> None:
    logger.info("%s %s", event, json.dumps(payload, sort_keys=True, default=str))


def _merge_pending_steps(
    pending_steps: list[str],
    new_steps: list[SpecialistType],
    *,
    completed_steps: list[str],
) -> list[str]:
    explicit_requested = {step.value for step in new_steps}
    merged: list[str] = []
    seen: set[str] = set()

    def add(step_name: str) -> None:
        if step_name in seen:
            return
        seen.add(step_name)
        merged.append(step_name)

    for step_name in pending_steps:
        if step_name and (step_name not in completed_steps or step_name in explicit_requested):
            add(step_name)

    for step in new_steps:
        step_name = step.value
        if not step_name:
            continue
        if step_name in seen:
            continue
        # Re-adding a completed step is only allowed when the controller
        # explicitly requests it through validation.next_steps.
        add(step_name)

    return merged


def _consume_current_step(
    state: OrchestratorState,
    *,
    validation: ControllerValidation,
    settings: Settings,
) -> dict[str, Any]:
    current_step = ""
    if state.get("current_step"):
        current_step = str(state["current_step"])
    elif state.get("pending_steps"):
        current_step = str(state["pending_steps"][0])

    pending = list(state.get("pending_steps", []) or [])
    completed = list(state.get("completed_steps", []) or [])

    if current_step:
        if pending and pending[0] == current_step:
            pending.pop(0)
        elif current_step in pending:
            pending.remove(current_step)
        if current_step not in completed:
            completed.append(current_step)

    if validation.next_steps:
        pending = _merge_pending_steps(
            pending,
            list(validation.next_steps),
            completed_steps=completed,
        )

    # Keep the workflow progressing only when the controller has a reason to.
    if validation.action == ControllerAction.REASON:
        needs_reasoning = True
        requires_clarification = False
    elif validation.action == ControllerAction.CLARIFY:
        needs_reasoning = False
        requires_clarification = True
    elif validation.action == ControllerAction.FINALIZE:
        needs_reasoning = False
        requires_clarification = False
        pending = []
    else:
        needs_reasoning = bool(validation.needs_reasoning or state.get("needs_reasoning", False))
        requires_clarification = bool(state.get("requires_clarification", False))

    controller_cycles = int(state.get("controller_cycles", 0) or 0) + 1
    specialist_executions = int(state.get("specialist_executions", 0) or 0)

    snapshot = {
        "current_step": "",
        "pending_steps": pending,
        "completed_steps": completed,
        "needs_reasoning": needs_reasoning,
        "requires_clarification": requires_clarification,
    }
    signature = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    previous_signature = str(state.get("last_progress_signature", "") or "")
    stalled = int(state.get("workflow_stall_count", 0) or 0)
    if signature == previous_signature:
        stalled += 1
    else:
        stalled = 0

    limit_hit = (
        controller_cycles >= settings.max_controller_cycles
        or specialist_executions >= settings.max_specialist_executions
        or stalled >= settings.workflow_stall_limit
    )

    if limit_hit:
        pending = []
        needs_reasoning = False
        requires_clarification = False

    updates: dict[str, Any] = {
        "current_step": "",
        "pending_steps": pending,
        "completed_steps": completed,
        "needs_reasoning": needs_reasoning,
        "requires_clarification": requires_clarification,
        "controller_cycles": controller_cycles,
        "specialist_executions": specialist_executions,
        "workflow_stall_count": stalled,
        "last_progress_signature": signature,
    }

    if limit_hit:
        updates["error"] = (
            "Workflow terminated safely after reaching a progress or execution limit."
        )
        updates["answer"] = updates["error"]
        updates["final_answer_ready"] = True

    return updates


def _step_update(state: OrchestratorState, step: SpecialistType) -> dict[str, Any]:
    return {
        "current_step": step.value,
        "specialist_executions": int(state.get("specialist_executions", 0) or 0) + 1,
    }


def _select_next_node(state: OrchestratorState) -> str:
    pending = list(state.get("pending_steps", []) or [])
    if pending:
        return pending[0]
    if state.get("needs_reasoning"):
        return "reasoning"
    if state.get("requires_clarification"):
        return "clarify"
    return "finalize"


def _update_used_models(state: OrchestratorState, model_name: str) -> list[str]:
    used = list(state.get("used_models", []) or [])
    if model_name not in used:
        used.append(model_name)
    return used


def _update_used_tools(state: OrchestratorState, tool_name: str) -> list[str]:
    used = list(state.get("used_tools", []) or [])
    if tool_name not in used:
        used.append(tool_name)
    return used


def make_prepare_node(settings: Settings):
    async def prepare_node(state: OrchestratorState) -> dict[str, Any]:
        messages = state.get("messages", []) or []
        user_text = last_user_text(messages)
        image_refs = collect_latest_message_images(messages, settings.vision_max_images)

        metadata = dict(state.get("metadata", {}) or {})
        metadata.setdefault("has_images", bool(image_refs))
        metadata.setdefault("user_text", user_text)

        return {
            "user_text": user_text,
            "has_images": bool(image_refs),
            "metadata": metadata,
            "used_models": list(state.get("used_models", []) or []),
            "used_tools": list(state.get("used_tools", []) or []),
            "messages": messages,
        }

    return prepare_node


def make_controller_plan_node(controller: ControllerEngine, settings: Settings):
    async def controller_plan_node(state: OrchestratorState) -> dict[str, Any]:
        stream = get_current_stream()
        if stream:
            await stream.controller_started(step="planning")

        plan = await controller.plan(state)
        route = plan_to_route(plan)
        pending_steps = [step.value for step in plan.execution_steps]

        if state.get("has_images") and settings.enable_vision:
            if SpecialistType.VISION.value not in pending_steps:
                pending_steps.insert(0, SpecialistType.VISION.value)
            plan.requires_vision = True
            if not plan.execution_steps or plan.execution_steps[0] != SpecialistType.VISION:
                plan.execution_steps = [SpecialistType.VISION] + [
                    step for step in plan.execution_steps if step != SpecialistType.VISION
                ]

        if stream:
            await stream.controller_plan(intent=plan.intent, steps=pending_steps)

        metadata = dict(state.get("metadata", {}) or {})
        metadata.update(
            {
                "controller_intent": plan.intent,
                "controller_complexity": plan.complexity,
                "controller_confidence": plan.confidence,
                "controller_requires_reasoning": plan.requires_reasoning,
                "controller_requires_clarification": plan.requires_clarification,
            }
        )

        _log_transition(
            "controller_plan",
            controller_decision=route.route.value,
            selected_next_node=_select_next_node(
                {
                    **state,
                    "pending_steps": pending_steps,
                    "needs_reasoning": bool(plan.requires_reasoning),
                    "requires_clarification": bool(plan.requires_clarification),
                    "current_step": "",
                }
            ),
            **_state_snapshot({
                **state,
                "pending_steps": pending_steps,
                "needs_reasoning": bool(plan.requires_reasoning),
                "requires_clarification": bool(plan.requires_clarification),
                "current_step": "",
            }),
        )

        return {
            "controller_plan": plan.model_dump(exclude_none=True),
            "route": route.model_dump(),
            "pending_steps": pending_steps,
            "completed_steps": list(state.get("completed_steps", []) or []),
            "current_step": "",
            "needs_reasoning": bool(plan.requires_reasoning),
            "requires_clarification": bool(plan.requires_clarification),
            "clarification_question": plan.clarification_question or "",
            "metadata": metadata,
            "used_models": _update_used_models(state, settings.controller_model),
        }

    return controller_plan_node


def make_vision_node(vision_pipeline: VisionPipeline, settings: Settings):
    async def vision_node(state: OrchestratorState) -> dict[str, Any]:
        updates = _step_update(state, SpecialistType.VISION)
        if not settings.enable_vision:
            _log_transition(
                "specialist_complete",
                specialist=SpecialistType.VISION.value,
                **_state_snapshot({**state, **updates}),
                selected_next_node="validate",
            )
            return updates

        stream = get_current_stream()
        image_refs = collect_latest_message_images(state.get("messages", []), settings.vision_max_images)
        if stream and image_refs:
            await stream.vision_started(image_count=len(image_refs))

        result = await vision_pipeline.process(state)

        if result is None:
            cleaned = strip_images_from_messages(state.get("messages", []))
            if stream:
                await stream.vision_finished(summary="No image attachments were found.")
            updates.update({"messages": cleaned})
            _log_transition(
                "specialist_complete",
                specialist=SpecialistType.VISION.value,
                **_state_snapshot({**state, **updates}),
                selected_next_node="validate",
            )
            return updates

        analysis = result.analysis
        vision_context = result.context_markdown

        if stream:
            await stream.vision_progress(
                message=analysis.summary[:200] or "Vision analysis completed.",
                data={
                    "task_type": analysis.task_type.value,
                    "confidence": analysis.confidence,
                    "cache_hit": result.cache_hit,
                },
            )
            await stream.vision_finished(summary=analysis.summary[:200])

        metadata = dict(state.get("metadata", {}) or {})
        metadata["vision_cache_hit"] = result.cache_hit
        metadata["vision_task_type"] = analysis.task_type.value

        return {
            **updates,
            "vision": analysis.model_dump(exclude_none=True),
            "vision_context": vision_context,
            "messages": result.cleaned_messages or state.get("messages", []),
            "metadata": metadata,
            "used_models": _update_used_models(state, analysis.source_model or settings.vision_model),
        }

    return vision_node


def make_knowledge_node(knowledge_client: KnowledgeClient, settings: Settings):
    async def knowledge_node(state: OrchestratorState) -> dict[str, Any]:
        updates = _step_update(state, SpecialistType.KNOWLEDGE)
        if not settings.enable_rag:
            _log_transition(
                "specialist_complete",
                specialist=SpecialistType.KNOWLEDGE.value,
                **_state_snapshot({**state, **updates}),
                selected_next_node="validate",
            )
            return updates

        stream = get_current_stream()
        query = last_user_text(state.get("messages", []))
        if not query.strip():
            _log_transition(
                "specialist_complete",
                specialist=SpecialistType.KNOWLEDGE.value,
                **_state_snapshot({**state, **updates}),
                selected_next_node="validate",
            )
            return updates

        if stream:
            await stream.knowledge_started(query=query[:200])

        result = await knowledge_client.retrieve(
            question=query,
            top_k=settings.knowledge_top_k,
            candidate_limit=settings.knowledge_candidate_limit,
            neighbor_window=settings.knowledge_neighbor_window,
        )

        if stream:
            sources = [f"{hit.repository}:{hit.path}" for hit in result.primary_hits[:3]]
            await stream.knowledge_finished(
                documents=len(result.primary_hits),
                sources=sources,
            )

        result_payload = {
            **updates,
            "knowledge_result": result.model_dump(exclude_none=True),
            "used_tools": _update_used_tools(state, "knowledge.retrieve"),
        }
        _log_transition(
            "specialist_complete",
            specialist=SpecialistType.KNOWLEDGE.value,
            **_state_snapshot({**state, **result_payload}),
            selected_next_node="validate",
        )
        return result_payload

    return knowledge_node


def _build_coder_prompt(state: OrchestratorState) -> list[ChatMessage]:
    user_text = last_user_text(state.get("messages", []))
    plan = _as_plan(state.get("controller_plan"))
    knowledge = _as_knowledge(state.get("knowledge_result"))
    vision = state.get("vision_context", "")
    validation = _as_validation(state.get("controller_validation"))

    context_parts = [
        "You are the coding specialist.",
        "Return STRICT JSON ONLY with this schema:",
        '{ "task": "...", "summary": "...", "code": "...", "files": [], "tests": [], "warnings": [], "confidence": 0.0 }',
        "",
        f"User request:\n{user_text}",
    ]
    if plan:
        context_parts.append(f"Controller intent: {plan.intent}")
        context_parts.append(f"Controller summary: {plan.summary}")
    if knowledge and knowledge.context:
        context_parts.append(f"Knowledge context:\n{knowledge.context}")
    if vision:
        context_parts.append(f"Vision context:\n{vision}")
    if validation:
        context_parts.append(f"Latest controller validation:\n{validation.model_dump_json(indent=2)}")

    return [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content="\n".join(context_parts).strip(),
        )
    ]


def make_coder_node(controller: ControllerEngine, settings: Settings):
    async def coder_node(state: OrchestratorState) -> dict[str, Any]:
        updates = _step_update(state, SpecialistType.CODER)
        stream = get_current_stream()
        if stream:
            await stream.code_started(model=settings.coder_model)

        messages = _build_coder_prompt(state)

        response = await controller.ollama.chat(
            model=settings.coder_model,
            messages=messages,
            temperature=0.15,
            max_tokens=settings.coder_max_tokens,
            stream=False,
            keep_alive=settings.controller_keep_alive,
        )

        parsed: dict[str, Any] = {}
        try:
            import json
            text = response.content.strip()
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(text[start : end + 1])
            else:
                parsed = json.loads(text)
        except Exception:
            parsed = {}

        coder_result = CoderResult(
            task=str(parsed.get("task") or "").strip(),
            summary=str(parsed.get("summary") or "").strip() or response.content[:400],
            code=str(parsed.get("code") or response.content).strip(),
            files=[str(item) for item in (parsed.get("files") or []) if str(item).strip()],
            tests=[str(item) for item in (parsed.get("tests") or []) if str(item).strip()],
            warnings=[str(item) for item in (parsed.get("warnings") or []) if str(item).strip()],
            confidence=float(parsed.get("confidence") or 0.0),
            raw_text=response.content,
        )

        if stream:
            await stream.code_finished(result=coder_result.summary[:500])

        result_payload = {
            **updates,
            "coder_result": coder_result.model_dump(exclude_none=True),
            "used_models": _update_used_models(state, settings.coder_model),
        }
        _log_transition(
            "specialist_complete",
            specialist=SpecialistType.CODER.value,
            **_state_snapshot({**state, **result_payload}),
            selected_next_node="validate",
        )
        return result_payload

    return coder_node


def make_tools_node(settings: Settings):
    async def tools_node(state: OrchestratorState) -> dict[str, Any]:
        updates = _step_update(state, SpecialistType.TOOLS)
        plan = _as_plan(state.get("controller_plan"))
        tool_requests = list(plan.tool_requests if plan else [])
        if not tool_requests:
            result_payload = {
                **updates,
                "tool_result": ToolResult(
                    tool_name="mcp",
                    status="skipped",
                    summary="No tool requests were produced by the controller.",
                    result={},
                    raw_text="",
                ).model_dump(exclude_none=True)
            }
            _log_transition(
                "specialist_complete",
                specialist=SpecialistType.TOOLS.value,
                **_state_snapshot({**state, **result_payload}),
                selected_next_node="validate",
            )
            return result_payload

        summaries: list[str] = []
        for request in tool_requests:
            summaries.append(
                f"{request.tool_name}: {request.description or 'no description'}"
            )

        result_payload = {
            **updates,
            "tool_result": ToolResult(
                tool_name="mcp",
                status="not_configured",
                summary="; ".join(summaries),
                result={
                    "tool_requests": [item.model_dump(exclude_none=True) for item in tool_requests],
                    "message": "MCP execution is not wired yet. The controller can still plan tool use.",
                },
                raw_text="",
            ).model_dump(exclude_none=True),
            "used_tools": _update_used_tools(state, "mcp.plan"),
        }
        _log_transition(
            "specialist_complete",
            specialist=SpecialistType.TOOLS.value,
            **_state_snapshot({**state, **result_payload}),
            selected_next_node="validate",
        )
        return result_payload

    return tools_node


def make_controller_validate_node(controller: ControllerEngine, settings: Settings):
    async def controller_validate_node(state: OrchestratorState) -> dict[str, Any]:
        step: SpecialistType | None = None
        if state.get("current_step"):
            try:
                step = SpecialistType(state["current_step"])
            except Exception:
                step = None
        elif state.get("pending_steps"):
            step = _step_from_pending(state.get("pending_steps"))

        validation = await controller.validate(state, last_step=step)

        updates = _consume_current_step(state, validation=validation, settings=settings)
        metadata = dict(state.get("metadata", {}) or {})
        metadata.update(
            {
                "validation_action": validation.action.value,
                "validation_confidence": validation.confidence,
                "validation_notes": validation.notes,
            }
        )
        updates["metadata"] = metadata
        updates["controller_validation"] = validation.model_dump(exclude_none=True)
        updates["used_models"] = _update_used_models(state, settings.controller_model)

        stream = get_current_stream()
        if stream:
            await stream.controller_validated(
                action=validation.action.value,
                issues=validation.issues,
            )

        selected_next_node = _select_next_node({**state, **updates})
        _log_transition(
            "controller_validated",
            controller_decision=validation.action.value,
            selected_next_node=selected_next_node,
            **_state_snapshot({**state, **updates}),
        )

        if validation.final_answer_ready:
            updates["final_answer_ready"] = True
        return updates

    return controller_validate_node


def make_reasoning_node(controller: ControllerEngine, settings: Settings):
    async def reasoning_node(state: OrchestratorState) -> dict[str, Any]:
        stream = get_current_stream()
        if stream:
            await stream.reasoning_started(model=settings.reasoning_model)

        content_parts: list[str] = []
        final_raw: dict[str, Any] = {}

        try:
            async for chunk in controller.ollama.stream_chat(
                model=settings.reasoning_model,
                messages=[
                    ChatMessage(
                        role=ChatRole.SYSTEM,
                        content="You are the deep reasoning model. Return plain text only.",
                    ),
                    ChatMessage(
                        role=ChatRole.SYSTEM,
                        metadata={"source": "structured_context"},
                        content=render_structured_context(
                            vision_context=state.get("vision_context", ""),
                            knowledge_result=_as_knowledge(state.get("knowledge_result")),
                            coder_result=_as_coder(state.get("coder_result")),
                            tool_result=_as_tool(state.get("tool_result")),
                            controller_plan=_as_plan(state.get("controller_plan")),
                            controller_validation=_as_validation(state.get("controller_validation")),
                        )
                        or "No structured context available.",
                    ),
                    ChatMessage(
                        role=ChatRole.USER,
                        content=last_user_text(state.get("messages", [])),
                    ),
                ],
                temperature=settings.reasoning_temperature,
                max_tokens=settings.reasoning_max_tokens,
                keep_alive=settings.reasoning_keep_alive,
            ):
                if chunk.content:
                    content_parts.append(chunk.content)
                    if stream:
                        await stream.reasoning_token(chunk.content)
                final_raw = chunk.raw or final_raw

            generation = ModelGenerationResponse(
                model=settings.reasoning_model,
                content="".join(content_parts).strip(),
                raw=final_raw,
            )

            if stream:
                await stream.reasoning_finished()

            return {
                "reasoning_result": generation.model_dump(exclude_none=True),
                "used_models": _update_used_models(state, settings.reasoning_model),
                "needs_reasoning": False,
                "final_answer_ready": True,
            }
        except Exception as exc:
            if stream:
                await stream.error(str(exc), stage="reasoning")
            raise

    return reasoning_node


def make_clarify_node():
    async def clarify_node(state: OrchestratorState) -> dict[str, Any]:
        plan = _as_plan(state.get("controller_plan"))
        answer = ""
        if plan and plan.clarification_question:
            answer = plan.clarification_question.strip()
        if not answer:
            answer = (
                "I need one more detail to route this cleanly. "
                "What exactly should I optimize for here: image analysis, code generation, knowledge lookup, or tool execution?"
            )

        assistant_message = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=answer,
            metadata={"route": "clarify"},
        )

        return {
            "answer": answer,
            "messages": [*state.get("messages", []), assistant_message.model_dump(exclude_none=True)],
            "final_answer_ready": True,
        }

    return clarify_node


def make_finalize_node(controller: ControllerEngine, settings: Settings):
    async def finalize_node(state: OrchestratorState) -> dict[str, Any]:
        stream = get_current_stream()

        existing_answer = str(state.get("answer", "") or "").strip()
        if existing_answer:
            answer = existing_answer
            model = str(state.get("metadata", {}).get("final_model") or settings.controller_model)
        else:
            if stream:
                await stream.llm_started(model=settings.controller_model)

            generation = await controller.finalize(state)
            model = generation.model
            answer = generation.content.strip()

            if stream:
                await stream.llm_finished()

        if not answer.strip():
            answer = (
                "I could not generate a complete answer for that request. "
                "Please try again with a little more detail."
            )

        assistant_message = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=answer,
            metadata={
                "model": model,
                "controller_plan": state.get("controller_plan"),
                "controller_validation": state.get("controller_validation"),
            },
        )

        metadata = dict(state.get("metadata", {}) or {})
        metadata.update(
            {
                "final_model": model,
                "final_answer_ready": True,
                "final_answer": answer,
            }
        )

        return {
            "answer": answer,
            "messages": [*state.get("messages", []), assistant_message.model_dump(exclude_none=True)],
            "metadata": metadata,
            "used_models": _update_used_models(state, model),
            "final_answer_ready": True,
        }

    return finalize_node
