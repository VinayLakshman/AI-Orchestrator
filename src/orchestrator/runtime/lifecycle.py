"""Maintained lifecycle boundary.

The implementation remains in ``model_lifecycle`` while the public runtime
composition code depends on this responsibility-focused facade. This keeps
the historical module importable during the lifecycle split.
"""

from .model_lifecycle import (
    LifecycleError,
    LifecycleState,
    ModelLifecycle,
    ModelRuntimeState,
    is_llm_container_owner,
)

__all__ = [
    "LifecycleError",
    "LifecycleState",
    "ModelLifecycle",
    "ModelRuntimeState",
    "is_llm_container_owner",
]
