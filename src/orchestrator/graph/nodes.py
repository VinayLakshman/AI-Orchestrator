from __future__ import annotations

from typing import Any

from ..clients import KnowledgeClient, OllamaClient
from ..router import RequestRouter
from ..schemas import ChatMessage, ChatRole, KnowledgeHit, RouteDecision, RouteType
from ..settings import Settings
from ..vision.pipeline import VisionPipeline
from ..vision.prompts import build_vision_injection_message
from ..vision.fetcher import strip_images_from_messages
from .prompts import (
    BASE_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    CLARIFY_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    TOOLS_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
)
from .state import OrchestratorState


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == ChatRole.USER.value and message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = str(part.get("text", "")).strip()
                        if text:
                            text_parts.append(text)
                return "\n".join(text_parts).strip()
            return str(content)
    return ""


def state_messages_to_chat_messages(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    return [ChatMessage.model_validate(message) for message in messages]


def normalize_knowledge(raw: list[dict[str, Any]]) -> list[KnowledgeHit]:
    return [KnowledgeHit.model_validate(item) for item in raw]


def render_knowledge_context(chunks: list[KnowledgeHit]) -> str:
    if not chunks:
        return ""

    lines: list[str] = ["Retrieved knowledge context:"]
    for idx, chunk in enumerate(chunks, start=1):
        score = f" (score={chunk.score:.3f})" if isinstance(chunk.score, float) else ""
        source = f" source={chunk.repository}" if chunk.repository else ""
        lines.append(f"[{idx}]{score}{source}")
        lines.append(chunk.content.strip())
        lines.append("")
    return "\n".join(lines).strip()


def choose_model(settings: Settings, decision: RouteDecision) -> str:
    if decision.route == RouteType.VISION:
        return settings.vision_model
    if decision.route == RouteType.CODE:
        return settings.coder_model
    return settings.general_model


def system_prompt_for_route(decision: RouteDecision) -> str:
    if decision.route == RouteType.VISION:
        return "\n\n".join([BASE_SYSTEM_PROMPT, VISION_SYSTEM_PROMPT])
    if decision.route == RouteType.CODE:
        return "\n\n".join([BASE_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT])
    if decision.route == RouteType.RAG:
        return "\n\n".join([BASE_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT])
    if decision.route == RouteType.TOOLS:
        return "\n\n".join([BASE_SYSTEM_PROMPT, TOOLS_SYSTEM_PROMPT])
    if decision.route == RouteType.CLARIFY:
        return "\n\n".join([BASE_SYSTEM_PROMPT, CLARIFY_SYSTEM_PROMPT])
    if decision.route == RouteType.MULTI_STEP:
        return BASE_SYSTEM_PROMPT
    return BASE_SYSTEM_PROMPT


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
            return {
                "knowledge": [],
                "knowledge_context": "",
                "retrieval_stats": {},
            }

        question = last_user_text(state.get("messages", []))
        result = await knowledge_client.retrieve(
            question=question,
            top_k=settings.knowledge_top_k,
            candidate_limit=settings.knowledge_candidate_limit,
            neighbor_window=settings.knowledge_neighbor_window,
        )

        hits = [*result.primary_hits, *result.expanded_hits]
        knowledge_context = result.context.strip()

        if not knowledge_context and hits:
            knowledge_context = render_knowledge_context(hits)

        return {
            "knowledge": [hit.model_dump(exclude_none=True) for hit in hits],
            "knowledge_context": knowledge_context,
            "retrieval_stats": {
                "question": result.question,
                "intent": result.intent,
                "embedding_time": result.embedding_time,
                "search_time": result.search_time,
                "rerank_time": result.rerank_time,
                "expansion_time": result.expansion_time,
                "total_time": result.total_time,
                "primary_hits": len(result.primary_hits),
                "expanded_hits": len(result.expanded_hits),
            },
            "used_tools": list(dict.fromkeys(state.get("used_tools", []) + ["knowledge_service"])),
            "metadata": {
                **state.get("metadata", {}),
                "knowledge_hits": len(hits),
                "knowledge_query": question,
                "knowledge_intent": result.intent,
                "knowledge_total_time": result.total_time,
            },
        }

    return retrieve_node


def make_generate_node(ollama_client: OllamaClient, settings: Settings):
    async def generate_node(state: OrchestratorState) -> dict[str, Any]:
        route_raw = state.get("route") or {}
        decision = RouteDecision.model_validate(route_raw)

        model = choose_model(settings, decision)
        system_prompt = system_prompt_for_route(decision)

        vision_context = (state.get("vision_context") or "").strip()
        knowledge_context = (state.get("knowledge_context") or "").strip()
        knowledge_hits = normalize_knowledge(state.get("knowledge", []))

        chat_messages = state_messages_to_chat_messages(state.get("messages", []))

        outgoing_messages: list[ChatMessage] = [
            ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)
        ]

        if vision_context:
            outgoing_messages.append(
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=build_vision_injection_message(
                        vision_context,
                        last_user_text(state.get("messages", [])),
                    ),
                    metadata={"source": "vision", "kind": "analysis_context"},
                )
            )

        if knowledge_context:
            outgoing_messages.append(
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=knowledge_context,
                    metadata={"source": "knowledge_service", "kind": "retrieval_context"},
                )
            )
        elif knowledge_hits:
            outgoing_messages.append(
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=render_knowledge_context(knowledge_hits),
                    metadata={"source": "knowledge_service", "kind": "retrieved_hits"},
                )
            )

        if decision.route == RouteType.TOOLS:
            outgoing_messages.append(
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=(
                        "If the user asked for action execution, explain the intended action "
                        "and whether the tool executor is available. Do not fabricate tool results."
                    ),
                )
            )

        outgoing_messages.extend(chat_messages)

        generation = await ollama_client.chat(
            model=model,
            messages=outgoing_messages,
            temperature=0.15 if decision.route in {RouteType.CODE, RouteType.RAG, RouteType.VISION} else 0.35,
            max_tokens=1400,
            stream=False,
        )

        assistant_message = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=generation.content.strip(),
            metadata={
                "model": model,
                "route": decision.route.value,
            },
        )

        return {
            "answer": assistant_message.content,
            "messages": [*state.get("messages", []), assistant_message.model_dump(exclude_none=True)],
            "used_models": list(dict.fromkeys(state.get("used_models", []) + [model])),
            "metadata": {
                **state.get("metadata", {}),
                "generation_model": model,
            },
        }

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