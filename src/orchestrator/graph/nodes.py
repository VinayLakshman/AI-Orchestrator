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
from ..schemas import ChatMessage, ChatRole, ModelGenerationResponse, RouteDecision, RouteType
from ..settings import Settings
from ..streaming.context import get_current_stream
from ..vision.fetcher import collect_latest_message_images, strip_images_from_messages
from ..vision.pipeline import VisionPipeline
from .state import OrchestratorState


def _knowledge_hit_sources(result: Any) -> list[str]:
    sources: list[str] = []
    if result is None:
        return sources

    for hit in getattr(result, "primary_hits", []) or []:
        repository = getattr(hit, "repository", "")
        path = getattr(hit, "path", "")
        if repository or path:
            sources.append(f"{repository}:{path}".strip(":"))
    return sources


def make_vision_node(vision_pipeline: VisionPipeline, settings: Settings):
    async def vision_node(state: OrchestratorState) -> dict[str, Any]:
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
                message="Vision analysis is complete and context has been prepared.",
                data={
                    "task": analysis.task_type.value,
                    "confidence": analysis.confidence,
                    "cache_hit": result.cache_hit,
                    "image_count": analysis.image_count,
                },
            )
            await stream.vision_finished(summary=analysis.summary)

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

        stream = get_current_stream()
        if stream:
            await stream.routing_started(query=user_text)

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

        if stream:
            await stream.routing_finished(route=decision.route.value, reason=decision.reason)

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
        stream = get_current_stream()
        if stream:
            await stream.knowledge_started(query=question)

        result = await knowledge_client.retrieve(
            question=question,
            top_k=settings.knowledge_top_k,
            candidate_limit=settings.knowledge_candidate_limit,
            neighbor_window=settings.knowledge_neighbor_window,
        )

        if stream:
            await stream.knowledge_progress(
                message=(
                    "Knowledge retrieval completed "
                    f"with {len(result.primary_hits)} primary hits."
                ),
                data={
                    "primary_hits": len(result.primary_hits),
                    "expanded_hits": len(result.expanded_hits),
                    "grounded": result.grounded,
                    "confidence": result.confidence,
                },
            )
            await stream.knowledge_finished(
                documents=len(result.primary_hits) + len(result.expanded_hits),
                sources=_knowledge_hit_sources(result),
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

        stream = get_current_stream()
        generation: ModelGenerationResponse

        if stream:
            if decision.route == RouteType.CODE or decision.needs_code:
                await stream.code_started(task="code generation")

            await stream.llm_started(model=model)

            content_parts: list[str] = []
            final_raw: dict[str, Any] = {}

            try:
                async for chunk in ollama_client.stream_chat(
                    model=model,
                    messages=outgoing,
                    temperature=temperature,
                    max_tokens=1400,
                ):
                    if chunk.content:
                        content_parts.append(chunk.content)
                        await stream.llm_token(chunk.content)
                    final_raw = chunk.raw or final_raw

                generation = ModelGenerationResponse(
                    model=model,
                    content="".join(content_parts),
                    raw=final_raw,
                )

                await stream.llm_finished()

                if decision.route == RouteType.CODE or decision.needs_code:
                    await stream.code_finished(result=generation.content[:500])
            except Exception as exc:
                await stream.error(str(exc), stage="generation")
                raise
        else:
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
        stream = get_current_stream()

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