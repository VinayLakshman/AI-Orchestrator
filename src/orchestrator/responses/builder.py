from __future__ import annotations

from typing import Any

from ..context.builder import last_user_text
from ..graph.state import OrchestratorState
from ..schemas import ChatMessage, ChatRole, KnowledgeRetrieveResponse


def build_retrieval_failure_response(
    state: OrchestratorState,
    knowledge_result: KnowledgeRetrieveResponse,
) -> dict[str, Any]:
    query = last_user_text(state.get("messages", []))
    retrieval_reason = str(
        knowledge_result.retrieval_reason
        or "Knowledge retrieval did not produce a grounded result"
    )
    confidence = knowledge_result.confidence

    answer = (
        "I couldn't answer this from your indexed knowledge base.\n\n"
        f"Query:\n{query}\n\n"
        f"Reason:\n{retrieval_reason}\n\n"
        "The orchestrator intentionally stopped before calling the language model "
        "because answering without grounded knowledge could produce hallucinated information."
    )

    assistant = ChatMessage(
        role=ChatRole.ASSISTANT,
        content=answer,
        metadata={
            "grounded": False,
            "retrieval_failed": True,
            "retrieval_reason": retrieval_reason,
            "confidence": confidence,
            "intent": knowledge_result.intent,
        },
    )

    return {
        "answer": answer,
        "messages": [
            *state.get("messages", []),
            assistant.model_dump(exclude_none=True),
        ],
        "used_models": state.get("used_models", []),
        "used_tools": state.get("used_tools", []),
        "metadata": {
            **state.get("metadata", {}),
            "rag_grounded": False,
            "retrieval_reason": retrieval_reason,
            "retrieval_confidence": confidence,
            "retrieval_intent": knowledge_result.intent,
            "knowledge_result": knowledge_result,
        },
    }


def build_generation_response(
    state: OrchestratorState,
    generation: Any,
    model: str,
    knowledge_result: KnowledgeRetrieveResponse | None = None,
) -> dict[str, Any]:
    content = generation.content.strip() if getattr(generation, "content", None) else str(generation)
    grounded = knowledge_result.grounded or False
    confidence = knowledge_result.confidence
    retrieval_reason = knowledge_result.retrieval_reason
    intent = knowledge_result.intent

    assistant = ChatMessage(
        role=ChatRole.ASSISTANT,
        content=content,
        metadata={
            "model": model,
            "route": state.get("route", {}).get("route") or state.get("route_name"),
            "grounded": grounded,
            "confidence": confidence,
            "retrieval_reason": retrieval_reason,
            "intent": intent,
        },
    )

    return {
        "answer": assistant.content,
        "messages": [
            *state.get("messages", []),
            assistant.model_dump(exclude_none=True),
        ],
        "used_models": list(dict.fromkeys(state.get("used_models", []) + [model])),
        "used_tools": state.get("used_tools", []),
        "metadata": {
            **state.get("metadata", {}),
            "generation_model": model,
            "rag_grounded": grounded,
            "retrieval_confidence": confidence,
            "retrieval_reason": retrieval_reason,
            "retrieval_intent": intent,
            "knowledge_result": knowledge_result,
        },
    }
