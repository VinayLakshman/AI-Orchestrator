from __future__ import annotations

from typing import Any

from ..clients import KnowledgeClient, OllamaClient
from ..context.builder import (
    build_generation_messages,
    last_user_text,
    resolve_system_prompt_for_route,
    select_model_for_route,
)
from ..responses.builder import build_generation_response, build_retrieval_failure_response
from ..router import RequestRouter
from ..schemas import ChatMessage, ChatRole, RouteDecision, RouteType
from ..settings import Settings
from ..vision.fetcher import strip_images_from_messages
from ..vision.pipeline import VisionPipeline
from .state import OrchestratorState


def make_vision_node(vision_pipeline: VisionPipeline, settings: Settings):
    async def vision_node(state: OrchestratorState) -> dict[str, Any]:
        result = await vision_pipeline.process(state)

        if result is None:
            cleaned = strip_images_from_messages(state.get("messages", []))
            return {"messages": cleaned}

        analysis = result.analysis
        vision_context = result.context_markdown

        return {
            "messages": result.cleaned_messages,
            "vision": analysis.model_dump(exclude_none=True),
            "vision_context": vision_context,
            "vision_task": analysis.task_type.value,
            "vision_confidence": analysis.confidence,
            "vision_image_hashes": result.image_hashes,
            "vision_cache_hit": result.cache_hit,
            "used_models": list(dict.fromkeys(state.get("used_models", []) + [settings.vision_model])),
            "metadata": {
                **state.get("metadata", {}),
                "vision_task": analysis.task_type.value,
                "vision_confidence": analysis.confidence,
                "vision_image_count": analysis.image_count,
                "vision_cache_hit": result.cache_hit,
                "vision_model": settings.vision_model,
            },
        }

    return vision_node


def make_route_node(router: RequestRouter, settings: Settings):
    async def route_node(state: OrchestratorState) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_text = last_user_text(messages)

        decision = await router.route(user_text)

        vision_context = state.get("vision_context") or ""
        if vision_context and decision.route == RouteType.GENERAL:
            decision = RouteDecision(
                route=RouteType.VISION,
                confidence=max(decision.confidence, state.get("vision_confidence") or 0.8),
                reason="Image analysis is available; routing to the vision generation path.",
                needs_vision=True,
                candidate_models=[settings.vision_model],
            )

        return {
            "route": decision.model_dump(),
            "route_name": decision.route.value,
            "metadata": {
                **state.get("metadata", {}),
                "user_text": user_text,
                "route_reason": decision.reason,
                "route_confidence": decision.confidence,
            },
        }

    return route_node


def make_retrieve_node(knowledge_client: KnowledgeClient, settings: Settings):
    async def retrieve_node(state: OrchestratorState) -> dict[str, Any]:
        route_raw = state.get("route") or {}
        decision = RouteDecision.model_validate(route_raw)

        if not decision.needs_rag:
            return {}

        question = last_user_text(state.get("messages", []))
        result = await knowledge_client.retrieve(
            question=question,
            top_k=settings.knowledge_top_k,
            candidate_limit=settings.knowledge_candidate_limit,
            neighbor_window=settings.knowledge_neighbor_window,
        )

        return {
            "knowledge_result": result,
            "used_tools": list(dict.fromkeys(state.get("used_tools", []) + ["knowledge_service"])),
            "metadata": {
                **state.get("metadata", {}),
                "knowledge_query": question,
                "knowledge_intent": result.intent,
                "knowledge_total_time": result.total_time,
            },
        }

    return retrieve_node


def make_generate_node(ollama_client: OllamaClient, settings: Settings):
    async def generate_node(state: OrchestratorState) -> dict[str, Any]:
        decision = RouteDecision.model_validate(state.get("route") or {})
        model = select_model_for_route(settings, decision)

        knowledge_result = state.get("knowledge_result")

        if decision.needs_rag:
            if knowledge_result is None:
                raise RuntimeError(
                    "BUG: retrieve node did not populate knowledge_result."
                )

        if (
            decision.needs_rag
            and knowledge_result is not None
            and not knowledge_result.grounded
        ):
            return build_retrieval_failure_response(
                state,
                knowledge_result,
            )

        outgoing = build_generation_messages(
            system_prompt=resolve_system_prompt_for_route(decision),
            messages=state.get("messages", []),
            vision_context=state.get("vision_context", ""),
            knowledge_result=knowledge_result,
            mcp_context=state.get("mcp_context", state.get("tool_context", "")),
            memory_context=state.get("memory_context", ""),
            latest_user_message=last_user_text(state.get("messages", [])),
        )

        temperature = (
            0.15
            if decision.route in (
                RouteType.RAG,
                RouteType.CODE,
                RouteType.VISION,
                RouteType.MULTI_STEP,
            )
            else 0.35
        )

        generation = await ollama_client.chat(
            model=model,
            messages=outgoing,
            temperature=temperature,
            max_tokens=1400,
            stream=False,
        )

        return build_generation_response(
            state,
            generation,
            model,
            knowledge_result,
        )

    return generate_node


def make_clarify_node():
    async def clarify_node(state: OrchestratorState) -> dict[str, Any]:
        route_raw = state.get("route") or {}
        decision = RouteDecision.model_validate(route_raw)

        answer = (
            "I need one more detail to route this cleanly. "
            "What exactly should I optimize for here: image analysis, code generation, knowledge lookup, or tool execution?"
        )

        if decision.route == RouteType.CLARIFY and decision.reason:
            answer = f"{decision.reason}\n\n{answer}"

        assistant_message = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=answer,
            metadata={"route": "clarify"},
        )

        return {
            "answer": answer,
            "messages": [*state.get("messages", []), assistant_message.model_dump(exclude_none=True)],
        }

    return clarify_node