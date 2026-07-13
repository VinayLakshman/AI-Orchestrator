from __future__ import annotations

from typing import Any

from ..clients import KnowledgeClient, OllamaClient
from ..context.builder import last_user_text, render_structured_context
from ..controller.engine import ControllerEngine, plan_to_route
from ..schemas import (
    ChatMessage,
    ChatRole,
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
from ..settings import Settings
from ..streaming.context import get_current_stream
from ..vision.fetcher import collect_latest_message_images, strip_images_from_messages
from ..vision.pipeline import VisionPipeline


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


def _pop_step(state: OrchestratorState, step: SpecialistType) -> dict[str, Any]:
    pending = list(state.get("pending_steps", []) or [])
    completed = list(state.get("completed_steps", []) or [])
    if pending and pending[0] == step.value:
        pending.pop(0)
    elif step.value in pending:
        pending.remove(step.value)
    if step.value not in completed:
        completed.append(step.value)
    return {
        "pending_steps": pending,
        "completed_steps": completed,
        "current_step": step.value,
    }


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

        return {
            "controller_plan": plan.model_dump(exclude_none=True),
            "route": route.model_dump(),
            "pending_steps": pending_steps,
            "completed_steps": [],
            "needs_reasoning": bool(plan.requires_reasoning),
            "requires_clarification": bool(plan.requires_clarification),
            "clarification_question": plan.clarification_question or "",
            "metadata": metadata,
            "used_models": _update_used_models(state, settings.controller_model),
        }

    return controller_plan_node


def make_vision_node(vision_pipeline: VisionPipeline, settings: Settings):
    async def vision_node(state: OrchestratorState) -> dict[str, Any]:
        if not settings.enable_vision:
            return {}

        stream = get_current_stream()
        image_refs = collect_latest_message_images(state.get("messages", []), settings.vision_max_images)
        if stream and image_refs:
            await stream.vision_started(image_count=len(image_refs))

        result = await vision_pipeline.process(state)

        if result is None:
            cleaned = strip_images_from_messages(state.get("messages", []))
            if stream:
                await stream.vision_finished(summary="No image attachments were found.")
            return {"messages": cleaned}

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
            "vision": analysis.model_dump(exclude_none=True),
            "vision_context": vision_context,
            "messages": result.cleaned_messages or state.get("messages", []),
            "metadata": metadata,
            "used_models": _update_used_models(state, analysis.source_model or settings.vision_model),
        }

    return vision_node


def make_knowledge_node(knowledge_client: KnowledgeClient, settings: Settings):
    async def knowledge_node(state: OrchestratorState) -> dict[str, Any]:
        if not settings.enable_rag:
            return {}

        stream = get_current_stream()
        query = last_user_text(state.get("messages", []))
        if not query.strip():
            return {}

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

        return {
            "knowledge_result": result.model_dump(exclude_none=True),
            "used_tools": _update_used_tools(state, "knowledge.retrieve"),
        }

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

        return {
            "coder_result": coder_result.model_dump(exclude_none=True),
            "used_models": _update_used_models(state, settings.coder_model),
        }

    return coder_node


def make_tools_node(settings: Settings):
    async def tools_node(state: OrchestratorState) -> dict[str, Any]:
        plan = _as_plan(state.get("controller_plan"))
        tool_requests = list(plan.tool_requests if plan else [])
        if not tool_requests:
            return {
                "tool_result": ToolResult(
                    tool_name="mcp",
                    status="skipped",
                    summary="No tool requests were produced by the controller.",
                    result={},
                    raw_text="",
                ).model_dump(exclude_none=True)
            }

        summaries: list[str] = []
        for request in tool_requests:
            summaries.append(
                f"{request.tool_name}: {request.description or 'no description'}"
            )

        return {
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

        pending = list(state.get("pending_steps", []) or [])
        if validation.next_steps:
            pending = [step.value for step in validation.next_steps] + pending
            dedup: list[str] = []
            for step_name in pending:
                if step_name not in dedup:
                    dedup.append(step_name)
            pending = dedup

        if validation.needs_reasoning:
            needs_reasoning = True
        else:
            needs_reasoning = bool(state.get("needs_reasoning", False))

        if validation.action == ControllerAction.CLARIFY:
            needs_reasoning = False

        metadata = dict(state.get("metadata", {}) or {})
        metadata.update(
            {
                "validation_action": validation.action.value,
                "validation_confidence": validation.confidence,
                "validation_notes": validation.notes,
            }
        )

        stream = get_current_stream()
        if stream:
            await stream.controller_validated(
                action=validation.action.value,
                issues=validation.issues,
            )

        updates = {
            "controller_validation": validation.model_dump(exclude_none=True),
            "pending_steps": pending,
            "needs_reasoning": needs_reasoning,
            "requires_clarification": validation.action == ControllerAction.CLARIFY,
            "metadata": metadata,
            "used_models": _update_used_models(state, settings.controller_model),
        }

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
        reasoning_result = state.get("reasoning_result")
        if isinstance(reasoning_result, dict):
            reasoning_result = ModelGenerationResponse.model_validate(reasoning_result)
        elif not isinstance(reasoning_result, ModelGenerationResponse):
            reasoning_result = None

        if reasoning_result is not None:
            answer = reasoning_result.content
            model = reasoning_result.model
        else:
            if stream:
                await stream.llm_started(model=settings.controller_model)

            generation = await controller.finalize(state)
            model = generation.model
            answer = generation.content.strip()

            if stream:
                await stream.llm_finished()

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