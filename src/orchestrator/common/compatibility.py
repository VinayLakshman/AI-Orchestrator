"""Explicit inventory of intentionally retained compatibility surfaces."""

from __future__ import annotations

from collections.abc import Iterable


COMPATIBILITY_MODULES = frozenset(
    {
        "orchestrator.graph.prompts",
        "orchestrator.common.utils",
        "orchestrator.models.ollama",
        "orchestrator.runtime.model_inference_guard",
        "orchestrator.runtime.model_lifecycle",
        "orchestrator.runtime.model_lifecycle.active_inference",
    }
)


def compatibility_exports(module: str, exports: Iterable[str]) -> list[str]:
    """Return a stable export list while documenting the compatibility owner."""
    if module not in COMPATIBILITY_MODULES:
        raise ValueError(f"{module!r} is not registered as a compatibility surface")
    return list(exports)


__all__ = ["COMPATIBILITY_MODULES", "compatibility_exports"]
