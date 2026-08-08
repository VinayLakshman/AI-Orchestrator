from __future__ import annotations

import json
from typing import Any

from orchestrator.controller.engine import ControllerEngine

from ..clients.knowledge import KnowledgeClient
from ..clients.searxng import normalize_query
from ..common.enums import ControllerAction, SpecialistType
from ..common.utils import _extract_json_object
from ..context.assembler import build_conversation
from ..context.parser import split_conversation
from ..models.chat import ChatMessage
from .prompts import (
    build_controller_plan_prompt,
    build_controller_validation_prompt,
)
from ..logging import get_logger
from ..models.evidence import (
    CodeEvidence,
    EvidenceLedger,
    ReasoningEvidence,
    RepositoryEvidence,
    ToolEvidence,
    VisionEvidence,
    WebEvidence,
)
from ..models.ollama import extract_assistant_text
from ..models.state import DebugState, OrchestratorState, ResponseState
from ..models.execution import (
    ExecutionState,
    RetryState,
    ValidationResult,
)
from ..settings import Settings
from ..specialists.web import WebSpecialist
from ..streaming.context import get_current_stream
from ..vision.pipeline import VisionPipeline

logger = get_logger(__name__)


def _request_user_text(
    state: OrchestratorState,
) -> str:
    return state.request.user_message


def _current_step_name(
    execution: ExecutionState,
) -> str:
    current = execution.runtime.current_specialist
    return current.value if current else ""


def _advance_runtime(
    execution: ExecutionState,
    specialist: SpecialistType,
    *,
    success: bool,
    error: str = "",
) -> ExecutionState:
    runtime = execution.runtime.model_copy(deep=True)
    if not runtime.completed or runtime.completed[-1] != specialist:
        runtime.completed.append(specialist)
    if not success:
        retry_state = runtime.retries.setdefault(specialist, RetryState())
        retry_state.attempts += 1
        retry_state.last_error = error
        runtime.metadata["last_error"] = error
    runtime.metadata["last_step"] = specialist.value
    runtime.metadata["last_status"] = "success" if success else "failed"
    runtime.current_index = min(int(runtime.current_index or 0) + 1, len(runtime.queue))
    runtime.current_specialist = runtime.queue[runtime.current_index] if runtime.current_index < len(runtime.queue) else None
    return execution.model_copy(update={"runtime": runtime})


def _rewind_runtime(execution: ExecutionState) -> ExecutionState:
    runtime = execution.runtime.model_copy(deep=True)
    if runtime.current_index > 0:
        runtime.current_index -= 1
    if runtime.completed:
        runtime.completed.pop()
    runtime.current_specialist = runtime.queue[runtime.current_index] if runtime.current_index < len(runtime.queue) else None
    runtime.metadata["last_status"] = "retry"
    return execution.model_copy(update={"runtime": runtime})


def _select_next_node(
    state: OrchestratorState,
) -> str:
    execution = state.execution
    validation = execution.validation
    runtime = execution.runtime

    if validation.retry:
        return _current_step_name(execution) or "finalize"

    if validation.action == ControllerAction.FINALIZE:
        return "finalize"

    if validation.action == ControllerAction.CLARIFY:
        return "clarify"

    if validation.action == ControllerAction.REASON:
        return "reasoning"

    current = runtime.current_specialist
    return current.value if current else "finalize"


def _state_snapshot(
    state: OrchestratorState,
) -> dict[str, Any]:
    execution = state.execution
    runtime = execution.runtime
    validation = execution.validation

    return {
        "classification": execution.plan.classification,
        "queue": [s.value for s in runtime.queue],
        "current_index": runtime.current_index,
        "current": runtime.current_specialist.value if runtime.current_specialist else "",
        "completed": [s.value for s in runtime.completed],
        "retry_counts": {
            specialist.value: retry.attempts
            for specialist, retry in runtime.retries.items()
        },
        "validation_action": validation.action.value,
        "validation_complete": validation.complete,
    }


def _log_transition(event: str, **payload: Any) -> None:
    logger.debug("%s %s", event, json.dumps(payload, sort_keys=True, default=str))


def _update_used_models(
    state: OrchestratorState,
    model_name: str,
 ) -> DebugState:
    if model_name and model_name not in state.debug.used_models:
        state.debug.used_models.append(model_name)
    return state.debug


def _update_used_tools(
    state: OrchestratorState,
    tool_name: str,
 ) -> DebugState:
    if tool_name and tool_name not in state.debug.used_tools:
        state.debug.used_tools.append(tool_name)
    return state.debug


def make_prepare_node(settings: Settings):
    async def prepare_node(state: OrchestratorState) -> OrchestratorState:
        state.execution = ExecutionState()
        state.execution.validation = ValidationResult()
        state.evidence = EvidenceLedger()
        state.response = ResponseState()
        state.debug = DebugState()
        return state

    return prepare_node


def make_controller_plan_node(controller: ControllerEngine, settings: Settings):
    async def controller_plan_node(state: OrchestratorState) -> OrchestratorState:
        stream = get_current_stream()

        if stream:
            await stream.controller_started(step="planning")

        plan = await controller.plan(state)

        if stream:
            await stream.controller_plan(
                intent=getattr(plan, "classification", "general"),
                steps=[step.value if hasattr(step, "value") else str(step) for step in (getattr(plan, "execution_queue", []) or [])],
            )

        execution = state.execution
        execution.plan = plan
        execution.validation = ValidationResult()
        execution.initialize()
        # DEBUG: trace planner->runtime queue transfer
        logger.debug(
            "DEBUG planner_to_runtime queue_after_initialize=%s plan_classification=%s plan_execution_queue=%s",
            [s.value for s in execution.runtime.queue],
            getattr(execution.plan, "classification", None),
            [s.value for s in (getattr(execution.plan, "execution_queue", []) or [])],
        )

        execution.runtime.metadata["controller_model"] = controller.models.controller().name
        state.execution = execution

        state.debug.planner_prompt = build_controller_plan_prompt()
        state.debug.planner_response = plan.model_dump(exclude_none=True)

        _log_transition(
            "controller_plan",
            **_state_snapshot(state),
        )

        return state

    return controller_plan_node


def make_vision_node(vision_pipeline: VisionPipeline, settings: Settings, model_lifecycle: Any):


    async def vision_node(state: OrchestratorState) -> OrchestratorState:
        execution = state.execution
        evidence = state.evidence

        execution.runtime.current_specialist = SpecialistType.VISION

        if not settings.enable_vision:
            evidence.vision = VisionEvidence(
                task=None,
                confidence=0.0,
                summary="Vision is disabled.",
                context="",
                observations=[],
                extracted_text="",
                detected_objects=[],
                metadata={"status": "failed", "reason": "vision_disabled"},
            )
            execution = _advance_runtime(execution, SpecialistType.VISION, success=False, error="vision_disabled")
            state.execution = execution
            _log_transition("specialist_complete", specialist=SpecialistType.VISION.value, **_state_snapshot(state))
            return state

        stream = get_current_stream()
        image_refs = state.request.images[: settings.vision_max_images]

        await model_lifecycle.ensure_warm("vision")

        if stream and image_refs:
            await stream.vision_started(image_count=len(image_refs))

        async with model_lifecycle.active_inference("vision"):
            result = await vision_pipeline.process(state)


        if result is None:
            evidence.vision = VisionEvidence(
                task=None,
                confidence=0.0,
                summary="No image attachments were found.",
                context="",
                observations=[],
                extracted_text="",
                detected_objects=[],
                metadata={"status": "failed", "reason": "no_images_found"},
            )
            if stream:
                await stream.vision_finished(summary="No image attachments were found.")
            execution = _advance_runtime(execution, SpecialistType.VISION, success=False, error="no_images_found")
            state.execution = execution
            _log_transition("specialist_complete", specialist=SpecialistType.VISION.value, **_state_snapshot(state))
            return state

        analysis = result.analysis
        summary = str(getattr(analysis, "summary", "") or "").strip()
        observations = _compact_lines(
            str(getattr(analysis, "observations", "") or getattr(analysis, "answer_context", "") or summary),
            max_items=8,
            max_chars=180,
        )
        evidence.vision = VisionEvidence(
            task=str(getattr(analysis, "task_type", "") and getattr(analysis.task_type, "value", analysis.task_type) or "").strip() or None,
            confidence=float(getattr(analysis, "confidence", 0.0) or 0.0),
            summary=summary,
            context=str(result.context_markdown or ""),
            observations=observations,
            extracted_text=str(getattr(analysis, "ocr", "") or getattr(analysis, "raw_text", "") or result.context_markdown or ""),
            detected_objects=[],
            metadata={
                "cache_hit": bool(result.cache_hit),
                "source_model": str(getattr(analysis, "source_model", "") or settings.vision_model),
                "image_count": int(getattr(analysis, "image_count", 0) or 0),
                "hashes": list(getattr(analysis, "hashes", []) or []),
            },
        )

        if stream:
            await stream.vision_progress(
                message=summary[:200] or "Vision analysis completed.",
                data={
                    "task_type": str(getattr(analysis, "task_type", "") and getattr(analysis.task_type, "value", analysis.task_type) or ""),
                    "confidence": float(getattr(analysis, "confidence", 0.0) or 0.0),
                    "cache_hit": bool(result.cache_hit),
                },
            )
            await stream.vision_finished(summary=summary[:200] or "Vision analysis completed.")

        model_lifecycle.touch("vision")
        await model_lifecycle.keep_warm("vision")

        execution = _advance_runtime(execution, SpecialistType.VISION, success=True)

        state.execution = execution
        _log_transition("specialist_complete", specialist=SpecialistType.VISION.value, **_state_snapshot(state))
        _update_used_models(state, str(getattr(analysis, "source_model", "") or settings.vision_model))
        return state

    return vision_node


def make_knowledge_node(knowledge_client: KnowledgeClient, settings: Settings):
    async def knowledge_node(state: OrchestratorState) -> OrchestratorState:
        execution = state.execution
        evidence = state.evidence
        execution.runtime.current_specialist = SpecialistType.KNOWLEDGE

        query = _request_user_text(state)

        if not settings.enable_rag:
            evidence.repository = RepositoryEvidence(
                repository=None,
                branch=None,
                commit=None,
                question=query,
                retrieval_reason="knowledge retrieval disabled",
                confidence=0.0,
                context="",
                hit_count=0,
                primary_hits=[],
                expanded_hits=[],
                metadata={"status": "failed", "reason": "rag_disabled"},
            )
            execution = _advance_runtime(execution, SpecialistType.KNOWLEDGE, success=False, error="rag_disabled")
            state.execution = execution
            _log_transition("specialist_complete", specialist=SpecialistType.KNOWLEDGE.value, **_state_snapshot(state))
            _update_used_tools(state, "knowledge.retrieve")
            return state

        if not query.strip():
            evidence.repository = RepositoryEvidence(
                repository=None,
                branch=None,
                commit=None,
                question=query,
                retrieval_reason="empty query",
                confidence=0.0,
                context="",
                hit_count=0,
                primary_hits=[],
                expanded_hits=[],
                metadata={"status": "failed", "reason": "empty_query"},
            )
            execution = _advance_runtime(execution, SpecialistType.KNOWLEDGE, success=False, error="empty_query")
            state.execution = execution
            _log_transition("specialist_complete", specialist=SpecialistType.KNOWLEDGE.value, **_state_snapshot(state))
            _update_used_tools(state, "knowledge.retrieve")
            return state

        stream = get_current_stream()
        if stream:
            await stream.knowledge_started(query=query[:200])

        result = await knowledge_client.retrieve(
            question=query,
            top_k=settings.knowledge_top_k,
            candidate_limit=settings.knowledge_candidate_limit,
            neighbor_window=settings.knowledge_neighbor_window,
        )

        if stream:
            sources = [f"{hit.repository}:{hit.path}" for hit in (result.primary_hits or [])[:3]]
            await stream.knowledge_finished(documents=len(result.primary_hits or []), sources=sources)

        primary_hits = [hit.model_dump(exclude_none=True) if hasattr(hit, "model_dump") else hit for hit in (result.primary_hits or [])]
        expanded_hits = [hit.model_dump(exclude_none=True) if hasattr(hit, "model_dump") else hit for hit in (result.expanded_hits or [])]

        evidence.repository = RepositoryEvidence(
            repository=(primary_hits[0].get("repository") if primary_hits else None),
            branch=(primary_hits[0].get("branch") if primary_hits else None),
            commit=(primary_hits[0].get("commit") if primary_hits else None),
            question=str(getattr(result, "question", "") or query),
            retrieval_reason=str(getattr(result, "retrieval_reason", "") or ""),
            confidence=float(getattr(result, "confidence", 0.0) or 0.0),
            context=str(getattr(result, "context", "") or ""),
            hit_count=len(primary_hits),
            primary_hits=primary_hits,
            expanded_hits=expanded_hits,
            metadata={
                "embedding_time": getattr(result, "embedding_time", None),
                "search_time": getattr(result, "search_time", None),
                "rerank_time": getattr(result, "rerank_time", None),
                "expansion_time": getattr(result, "expansion_time", None),
                "total_time": getattr(result, "total_time", None),
                "grounded": bool(getattr(result, "grounded", False)),
            },
        )

        execution = _advance_runtime(execution, SpecialistType.KNOWLEDGE, success=True)
        state.execution = execution
        _log_transition("specialist_complete", specialist=SpecialistType.KNOWLEDGE.value, **_state_snapshot(state))
        _update_used_tools(state, "knowledge.retrieve")
        return state

    return knowledge_node


def _summarize_web_results(results: list[Any]) -> str:
    parts: list[str] = []

    for item in results[:4]:
        title = str(getattr(item, "title", "") or "").strip()
        snippet = str(getattr(item, "snippet", "") or "").strip()

        if title and snippet:
            parts.append(f"{title}: {snippet}")
        elif title:
            parts.append(title)
        elif snippet:
            parts.append(snippet)

    return " | ".join(parts)


def make_web_node(web_specialist: WebSpecialist, settings: Settings):
    async def web_node(state: OrchestratorState) -> OrchestratorState:
        execution = state.execution
        evidence = state.evidence
        execution.runtime.current_specialist = SpecialistType.WEB

        query = normalize_query(_request_user_text(state))

        if not settings.web_search_enabled or not query:
            evidence.web = WebEvidence(
                query=query,
                confidence=0.0,
                summary="web search disabled or empty query",
                results=[],
                snippets=[],
                urls=[],
                metadata={
                    "status": "failed",
                    "reason": "web_disabled" if not settings.web_search_enabled else "empty_query",
                },
            )
            execution = _advance_runtime(execution, SpecialistType.WEB, success=False, error="web_disabled_or_empty_query")
            state.execution = execution
            _log_transition("specialist_complete", specialist=SpecialistType.WEB.value, **_state_snapshot(state))
            return state

        cached = evidence.web if evidence.web and evidence.web.query == query else None
        stream = get_current_stream()

        if cached is not None:
            result = cached
        else:
            if stream:
                await stream.web_search_started(query=query[:200])
            logger.info("web_search_started query=%r", query[:200])
            result = await web_specialist.retrieve(
                query,
                cached=None,
                max_results=settings.web_search_max_results,
            )
            if stream:
                await stream.web_search_processing(results=len(result.results or []))
            logger.info(
                "web_search_completed search_duration_ms=%d results_returned=%d",
                int(getattr(result, "search_time_ms", 0) or 0),
                len(result.results or []),
            )
            if stream:
                await stream.web_search_finished(
                    results=len(result.results or []),
                    search_time_ms=int(getattr(result, "search_time_ms", 0) or 0),
                )

        results = list(getattr(result, "results", []) or [])
        evidence.web = WebEvidence(
            query=str(getattr(result, "query", "") or query),
            confidence=0.0,
            summary=_summarize_web_results(results),
            results=[item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item for item in results],
            snippets=[
                str(item.snippet).strip()
                for item in results
                if item.snippet
            ],
            urls=[
                str(item.url).strip()
                for item in results
                if item.url
            ],
            metadata={
                "search_time_ms": int(getattr(result, "search_time_ms", 0) or 0),
                "error": str(getattr(result, "error", "") or ""),
                "status": "success" if results else "empty",
            },
        )

        execution = _advance_runtime(execution, SpecialistType.WEB, success=True)
        state.execution = execution
        _log_transition("specialist_complete", specialist=SpecialistType.WEB.value, **_state_snapshot(state))
        _update_used_tools(state, "web.search")
        return state

    return web_node


def _build_coder_prompt(state: OrchestratorState) -> list[ChatMessage]:
    execution = state.execution
    evidence = state.evidence

    structured_sections: list[str] = []

    classification = execution.plan.classification
    if classification:
        structured_sections.append(f"Classification: {classification}")

    if evidence.repository and evidence.repository.context:
        structured_sections.append(f"Knowledge context:\n{evidence.repository.context}")

    if evidence.web and evidence.web.summary:
        structured_sections.append(f"Web summary:\n{evidence.web.summary}")

    if evidence.vision and evidence.vision.context:
        structured_sections.append(f"Vision context:\n{evidence.vision.context}")

    if evidence.reasoning and evidence.reasoning.summary:
        structured_sections.append(
            "Reasoning conclusions:\n" + "\n".join(evidence.reasoning.conclusions or [evidence.reasoning.summary])
        )

    system_prompt = (
        "You are the coding specialist.\n"
        "Return STRICT JSON ONLY with this schema:\n"
        '{ "task": "...", "summary": "...", "code": "...", "files": [], "tests": [], "warnings": [], "confidence": 0.0 }'
    )

    return build_conversation(
        system_prompt=system_prompt,
        structured_context="\n\n".join(structured_sections),
        history=_coder_history(state),
        latest_user_message=_request_user_text(state),
    )


def _coder_history(state: OrchestratorState) -> list[ChatMessage]:
    """Recover conversation history for the coder.

    The coder is a specialist operating inside the orchestration graph. It
    receives the full request conversation; the request's own latest user
    message is supplied as the actual user message, so everything before it is
    treated purely as history.
    """
    try:
        history_messages, _, _ = split_conversation(state.request.messages)
        return history_messages
    except ValueError:
        return []


def make_coder_node(controller: ControllerEngine, settings: Settings, model_lifecycle: Any):


    async def coder_node(state: OrchestratorState) -> OrchestratorState:
        execution = state.execution
        evidence = state.evidence
        execution.runtime.current_specialist = SpecialistType.CODER

        # Model lifecycle: warm/cached residency is handled here.
        await model_lifecycle.ensure_warm("coder")

        stream = get_current_stream()
        if stream:
            await stream.code_started(model=settings.coder_model)

        messages = _build_coder_prompt(state)

        async with model_lifecycle.active_inference("coder"):
            response = await controller.models.client("coder").chat(
                model=settings.coder_model,
                messages=messages,
                temperature=0.15,
                max_tokens=settings.coder_max_tokens,
                stream=False,
                keep_alive=settings.controller_keep_alive,
            )

        text = extract_assistant_text(response.content) or extract_assistant_text(response.raw) or ""
        parsed = _extract_json_object(text)
        if not isinstance(parsed, dict):
            parsed = {}

        code = str(parsed.get("code") or text).strip()
        explanation = str(parsed.get("summary") or "").strip() or code[:400]
        files = [str(item).strip() for item in (parsed.get("files") or []) if str(item).strip()]

        if stream:
            await stream.code_finished(result=explanation[:500] or code[:500])

        evidence.code = CodeEvidence(
            language=str(parsed.get("language") or "").strip() or None,
            task=str(parsed.get("task") or "").strip(),
            summary=explanation,
            generated_code=code,
            explanation=explanation,
            files=files,
            tests=[str(item) for item in (parsed.get("tests") or [])],
            warnings=[str(item) for item in (parsed.get("warnings") or [])],
            confidence=float(parsed.get("confidence") or 0.0),
            metadata={
                "model": settings.coder_model,
            },
        )

        model_lifecycle.touch("coder")
        model_lifecycle.keep_warm("coder")
        execution = _advance_runtime(execution, SpecialistType.CODER, success=bool(code or explanation))

        state.execution = execution
        _log_transition("specialist_complete", specialist=SpecialistType.CODER.value, **_state_snapshot(state))
        _update_used_models(state, settings.coder_model)
        return state

    return coder_node


def make_tools_node(settings: Settings):
    async def tools_node(state: OrchestratorState) -> OrchestratorState:
        execution = state.execution
        evidence = state.evidence
        execution.runtime.current_specialist = SpecialistType.TOOLS

        plan = execution.plan
        tool_requests = list(getattr(plan, "tool_requests", []) or [])

        if not tool_requests:
            evidence.tools = ToolEvidence(
                executions=[],
                metadata={
                    "status": "skipped",
                    "message": "No tool requests were produced by the controller.",
                },
            )
            execution = _advance_runtime(execution, SpecialistType.TOOLS, success=False, error="no_tool_requests")
            state.execution = execution
            _log_transition("specialist_complete", specialist=SpecialistType.TOOLS.value, **_state_snapshot(state))
            return state

        executions: list[dict[str, Any]] = []
        for request in tool_requests:
            req = request.model_dump(exclude_none=True) if hasattr(request, "model_dump") else dict(request)
            executions.append(
                {
                    "tool_name": str(req.get("tool_name") or "tool"),
                    "success": False,
                    "inputs": dict(req),
                    "outputs": {
                        "status": "not_configured",
                        "message": "MCP execution is not wired yet.",
                    },
                    "duration_ms": None,
                }
            )

        evidence.tools = ToolEvidence(
            executions=executions,
            metadata={
                "status": "planned",
                "message": "Controller produced tool requests. Execution is not implemented yet.",
            },
        )

        execution = _advance_runtime(execution, SpecialistType.TOOLS, success=True)
        state.execution = execution
        _log_transition("specialist_complete", specialist=SpecialistType.TOOLS.value, **_state_snapshot(state))
        _update_used_tools(state, "mcp.plan")
        return state

    return tools_node


def make_controller_validate_node(controller: ControllerEngine, settings: Settings):
    async def controller_validate_node(state: OrchestratorState) -> OrchestratorState:
        execution = state.execution
        last_step = None
        completed = list(execution.runtime.completed)
        if completed:
            last_step = completed[-1]
        if last_step is None:
            last_step = _current_step_name(execution)
        if isinstance(last_step, str):
            try:
                last_step = SpecialistType(last_step)
            except Exception:
                last_step = None

        validation = await controller.validate(state, last_step=last_step)

        if validation.retry:
            execution = _rewind_runtime(execution)

        execution.validation = validation
        execution.runtime.metadata["validation_action"] = validation.action.value if hasattr(validation.action, "value") else str(validation.action)
        execution.runtime.metadata["validation_confidence"] = float(validation.confidence or 0.0)
        execution.runtime.metadata["validation_summary"] = validation.summary
        execution.runtime.metadata["controller_model"] = controller.models.controller().name
        state.execution = execution
        state.debug.validator_prompt = build_controller_validation_prompt()
        state.debug.validator_response = validation.model_dump(exclude_none=True)

        stream = get_current_stream()
        if stream:
            await stream.controller_validated(
                action=validation.action.value if hasattr(validation.action, "value") else str(validation.action),
                issues=list(validation.issues or []),
            )

        selected_next_node = _select_next_node(state)
        _log_transition(
            "controller_validated",
            controller_decision=selected_next_node,
            selected_next_node=selected_next_node,
            **_state_snapshot(state),
        )

        _update_used_models(state, settings.controller_model)
        return state

    return controller_validate_node


def make_reasoning_node(controller: ControllerEngine, settings: Settings, model_lifecycle: Any):


    async def reasoning_node(state: OrchestratorState) -> OrchestratorState:
        execution = state.execution
        evidence = state.evidence

        # Model lifecycle: warm/cached residency is handled here.
        await model_lifecycle.ensure_warm("reasoning")

        stream = get_current_stream()
        if stream:
            await stream.reasoning_started(model=settings.reasoning_model)

        async with model_lifecycle.active_inference("reasoning"):
            generation = await controller.reason(state)


        summary = str(generation.content or generation.raw or "").strip()
        if not summary:
            summary = "No reasoning output produced."

        evidence.reasoning = ReasoningEvidence(
            summary=summary,
            conclusions=_compact_lines(summary, max_items=8, max_chars=220),
            assumptions=[],
            metadata={
                "model": settings.reasoning_model,
                "source": "controller.reason",
            },
        )

        runtime = execution.runtime.model_copy(deep=True)
        current_step = None
        if runtime.current_index < len(runtime.queue):
            current_step = runtime.queue[runtime.current_index]
        if current_step == SpecialistType.REASONING:
            execution = _advance_runtime(execution, SpecialistType.REASONING, success=True)
        else:
            if SpecialistType.REASONING not in runtime.completed:
                runtime.completed.append(SpecialistType.REASONING)
            runtime.current_specialist = None
            execution = execution.model_copy(update={"runtime": runtime})

        model_lifecycle.touch("reasoning")
        await model_lifecycle.keep_warm("reasoning")

        state.execution = execution
        if stream:
            await stream.reasoning_finished()

        _log_transition("specialist_complete", specialist=SpecialistType.REASONING.value, **_state_snapshot(state))

        _update_used_models(state, settings.reasoning_model)
        return state

    return reasoning_node


def _compact_lines(text: str | None, *, max_items: int, max_chars: int) -> list[str]:
    lines = [
        _truncate(line.strip(" -•\t"), max_chars)
        for line in str(text or "").splitlines()
        if line.strip()
    ]
    deduped, _ = _dedupe_text_items(lines)
    return deduped[:max_items]


def _dedupe_text_items(values: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    deduped: list[str] = []
    removed = 0
    for value in values:
        item = _normalize_text(value)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(item)
    return deduped, removed


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").split()).strip()


def _truncate(text: str, limit: int = 220) -> str:
    cleaned = _normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def make_clarify_node():
    async def clarify_node(state: OrchestratorState) -> OrchestratorState:
        execution = state.execution
        response = state.response

        validation = execution.validation
        answer = str(validation.summary or validation.reason or "").strip()
        if not answer:
            answer = (
                "I need one more detail to route this cleanly. "
                "What exactly should I optimize for here: image analysis, code generation, knowledge lookup, or tool execution?"
            )

        response.final_response = answer
        response.finish_reason = "clarify"
        response.metadata["route"] = "clarify"

        current_step = None
        runtime = execution.runtime.model_copy(deep=True)
        if runtime.current_index < len(runtime.queue):
            current_step = runtime.queue[runtime.current_index]
        if current_step == SpecialistType.CLARIFY:
            execution = _advance_runtime(execution, SpecialistType.CLARIFY, success=True)
        else:
            if SpecialistType.CLARIFY not in runtime.completed:
                runtime.completed.append(SpecialistType.CLARIFY)
            runtime.current_specialist = None
            execution = execution.model_copy(update={"runtime": runtime})

        state.execution = execution
        return state

    return clarify_node


def make_finalize_node(controller: ControllerEngine, settings: Settings):
    async def finalize_node(state: OrchestratorState) -> OrchestratorState:
        response = state.response
        stream = get_current_stream()

        existing_answer = str(response.final_response or "").strip()
        if existing_answer:
            answer = existing_answer
            model = str(response.metadata.get("final_model") or settings.controller_model)
        else:
            generation = await controller.finalize(state, publisher=stream)
            model = str(generation.model or settings.controller_model)
            answer = str(extract_assistant_text(generation.content) or extract_assistant_text(generation.raw) or "").strip()

        if not answer.strip():
            answer = "I could not generate a complete answer for that request. Please try again with a little more detail."

        response.final_response = answer
        response.finish_reason = "stop"
        response.metadata["final_model"] = model
        response.metadata["final_answer_ready"] = True
        response.metadata["final_answer"] = answer

        _update_used_models(state, model)
        return state

    return finalize_node
