from __future__ import annotations

"""Internal helper types for ModelLifecycle.

This module is intentionally small to keep lifecycle logic readable.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_lifecycle import ModelLifecycle


@dataclass(slots=True)
class _InferenceGuard:
    lifecycle: "ModelLifecycle"
    role: str

    async def __aenter__(self) -> "_InferenceGuard":
        await self.lifecycle._begin_inference(self.role)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.lifecycle._end_inference(self.role)

