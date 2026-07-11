from __future__ import annotations

from typing import Any

from ..context.builder import last_user_text
from ..context.validator import RetrievalValidationResult
from ..graph.state import OrchestratorState
from ..schemas import ChatMessage, ChatRole


def build_retrieval_failure_response(
    state: OrchestratorState,
    validation: RetrievalValidationResult,
) -> dict[str, Any]:
    retrieval_stats = validation.metadata or {}

    query = retrieval_stats.get("question") or retrieval_stats.get("query") or last_user_text(state.get("messages", []))

    answer = (
        "I couldn't answer this from your indexed knowledge base.\n\n"
        f"Query:\n{query}\n\n"
        f"Reason:\n{validation.reason}\n\n"
        "The orchestrator intentionally stopped before calling the language model "
        "because answering without grounded knowledge could produce hallucinated information."
    )

    assistant = ChatMessage(
        role=ChatRole.ASSISTANT,
        content=answer,
        metadata={
            "grounded": False,
            "retrieval_failed": True,
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
            "retrieval_reason": validation.reason,
            "retrieval_score": validation.score,
            "retrieval_stats": retrieval_stats,
        },
    }


def build_generation_response(
    state: OrchestratorState,
    generation: Any,
    model: str,
    validation: RetrievalValidationResult | None = None,
) -> dict[str, Any]:
    assistant = ChatMessage(
        role=ChatRole.ASSISTANT,
        content=generation.content.strip(),
        metadata={
            "model": model,
            "route": state.get("route", {}).get("route") or state.get("route_name"),
            "grounded": validation.grounded if validation is not None else None,
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
            "rag_grounded": validation.grounded if validation is not None else None,
            "retrieval_stats": validation.metadata if validation is not None else {},
        },
    }
