"""Conversation history assembly driven by a token budget.

This module is the single authoritative, reusable component for building
conversation history from a complete set of OpenWebUI-style messages.

It intentionally owns ONLY the history-building responsibility:

- input:  complete conversation messages + a maximum token budget
- output: trimmed conversation history, chronological order preserved

Design guarantees:

- deterministic (same input + budget => identical output)
- O(n): a single backward pass over the messages
- newest messages always win (oldest are discarded first)
- never reorders, never summarizes, never mutates message content
- preserves every message role (system, user, assistant, tool, function,
  developer, and any unknown future role)

The class-based API is designed so future orchestration features (Conversation
State, Evidence Ledger, Active Resources, Planner Notes, ...) can prepend
structured context before conversation history without architectural changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..logging import get_logger
from ..models.chat import ChatMessage

logger = get_logger(__name__)

# Default history budget. Treat strictly as the conversation-history budget,
# NOT the total model context budget. Overridable per call and via settings.
DEFAULT_HISTORY_TOKEN_BUDGET = 12000


def estimate_text_tokens(text: str | None) -> int:
    """Estimate the number of tokens in ``text`` with a light, deterministic
    approximation. This mirrors the existing project-wide estimator (word
    count) so no heavyweight tokenizer dependency is introduced."""
    value = (text or "").strip()
    if not value:
        return 0
    return max(1, len(re.findall(r"\S+", value)))


def _message_role(message: dict[str, Any] | ChatMessage) -> str:
    if isinstance(message, ChatMessage):
        return str(message.role.value)
    return str(message.get("role") or "")


def _message_content_text(message: dict[str, Any] | ChatMessage) -> str:
    content: Any
    if isinstance(message, ChatMessage):
        content = message.content
    else:
        content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                for key in ("text", "content", "url"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break
        return "\n".join(parts).strip()

    if content is None:
        return ""

    return str(content).strip()


def _message_token_count(message: dict[str, Any] | ChatMessage) -> int:
    return estimate_text_tokens(_message_content_text(message))


@dataclass(slots=True)
class ConversationContextInfo:
    """Metadata describing how conversation history was trimmed."""

    messages_examined: int = 0
    messages_included: int = 0
    estimated_tokens: int = 0
    budget: int = 0
    truncated: bool = False
    # True only when the newest user message individually exceeded the budget
    # and was retained anyway (safety rule). The estimated token count may then
    # exceed the budget.
    budget_breached_by_current_user: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages_examined": self.messages_examined,
            "messages_included": self.messages_included,
            "estimated_tokens": self.estimated_tokens,
            "budget": self.budget,
            "truncated": self.truncated,
            "budget_breached_by_current_user": self.budget_breached_by_current_user,
        }


class ConversationContextBuilder:
    """Deterministic, token-budget-driven conversation history builder.

    The builder walks the conversation history starting from the newest message
    and accumulates messages until adding another would exceed the configured
    token budget. The accepted set is then reversed so chronological order is
    preserved.

    Safety rule: the newest user message is ALWAYS retained. If it individually
    exceeds the budget, it is still included and ``truncated`` is set to True
    (the returned estimated token count may then exceed the budget).
    """

    def __init__(self, token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET) -> None:
        self.token_budget = max(0, int(token_budget))

    # -- public API ---------------------------------------------------------

    def build(
        self,
        messages: Iterable[dict[str, Any] | ChatMessage] | None,
        *,
        exclude_roles: Iterable[str] | None = None,
    ) -> tuple[list[ChatMessage], ConversationContextInfo]:
        """Build trimmed, chronological history from ``messages``.

        ``exclude_roles`` optionally drops specific roles (e.g. SYSTEM) from
        history before budgeting. This is a caller-level filter, not a content
        modification; the builder still preserves every remaining role.
        """
        excluded = set(exclude_roles or ())
        raw = list(messages or [])

        # Pre-filter roles the caller wants excluded (e.g. SYSTEM messages that
        # the assembler will re-emit itself). This preserves the remaining
        # roles and their original relative order.
        candidates = [
            message
            for message in raw
            if _message_role(message) not in excluded
        ]

        info = ConversationContextInfo(
            messages_examined=len(raw),
            budget=self.token_budget,
        )

        # Newest-message-first selection in a single backward pass (O(n)).
        selected: list[dict[str, Any] | ChatMessage] = []
        running_tokens = 0
        truncated = False

        for message in reversed(candidates):
            message_tokens = _message_token_count(message)
            would_exceed = running_tokens + message_tokens > self.token_budget

            # Always keep the single newest message (which, after the optional
            # SYSTEM filter, is either the latest non-SYSTEM message or the
            # newest retained message). This is the safety rule: do not drop the
            # most recent content purely because of the history budget.
            is_newest = not selected

            if is_newest:
                selected.append(message)
                running_tokens += message_tokens
                if message_tokens > self.token_budget:
                    truncated = True
                    info.budget_breached_by_current_user = True
                continue

            if would_exceed:
                # Stop accumulating: adding this older message would exceed the
                # budget. Everything older is discarded.
                truncated = True
                break

            selected.append(message)
            running_tokens += message_tokens

        # Reverse back into chronological order.
        selected.reverse()

        history_chat_messages = [
            message
            if isinstance(message, ChatMessage)
            else ChatMessage.model_validate(message)
            for message in selected
        ]

        info.messages_included = len(history_chat_messages)
        info.estimated_tokens = running_tokens
        info.truncated = truncated or running_tokens > self.token_budget

        logger.debug(
            "ConversationContextBuilder messages_examined=%s messages_included=%s "
            "estimated_tokens=%s budget=%s truncated=%s",
            info.messages_examined,
            info.messages_included,
            info.estimated_tokens,
            info.budget,
            info.truncated,
        )

        return history_chat_messages, info

    # -- convenience / future-provenance ------------------------------------

    def build_history(
        self,
        messages: Iterable[dict[str, Any] | ChatMessage] | None,
        *,
        exclude_roles: Iterable[str] | None = None,
    ) -> list[ChatMessage]:
        """Return only the trimmed history (convenience wrapper)."""
        history, _ = self.build(
            messages,
            exclude_roles=exclude_roles,
        )
        return history

