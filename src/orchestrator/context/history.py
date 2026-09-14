"""Public conversation-history boundary.

The implementation remains in ``conversation_builder`` for import
compatibility, while orchestration code uses this responsibility-focused
module.
"""

from .conversation_builder import (
    DEFAULT_HISTORY_TOKEN_BUDGET,
    ConversationContextBuilder,
    ConversationContextInfo,
    estimate_text_tokens,
)

__all__ = [
    "DEFAULT_HISTORY_TOKEN_BUDGET",
    "ConversationContextBuilder",
    "ConversationContextInfo",
    "estimate_text_tokens",
]
