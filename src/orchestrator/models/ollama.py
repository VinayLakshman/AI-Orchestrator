"""Backward-compatible imports for the former provider-specific module."""

from ..common.compatibility import compatibility_exports
from .generation import (
    ModelGenerationResponse,
    extract_assistant_text,
    normalize_generation_response,
)

__all__ = compatibility_exports(
    "orchestrator.models.ollama",
    ("ModelGenerationResponse", "extract_assistant_text", "normalize_generation_response"),
)
