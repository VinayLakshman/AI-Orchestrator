from __future__ import annotations

"""Static model metadata for residency policies.

Single source of truth for:
- role -> model name mapping (resolved elsewhere)
- keep-alive windows
- warm/eviction capabilities
- priority used by the lifecycle manager

This module must contain NO runtime state.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ModelPolicy:
    """Policy + hints used by the lifecycle manager."""

    role: str
    # Lower numbers indicate *less* important models.
    # The eviction algorithm evicts lowest priority first.
    priority: int

    # Keep the model warm (and eligible for eviction based on keep-alive expiry).
    keep_alive_seconds: int

    # Whether the model can be evicted/unloaded after becoming IDLE.
    can_evict: bool

    # Whether lifecycle manager may start warming in the background.
    preload_enabled: bool


# Controller is expected to remain resident.
CONTROLLER: Final[ModelPolicy] = ModelPolicy(
    role="controller",
    priority=100,
    keep_alive_seconds=3600,
    can_evict=False,
    preload_enabled=False,
)

# Specialist defaults below are overridden at runtime from Settings keep-alive
# configuration inside ModelLifecycle.
REASONING: Final[ModelPolicy] = ModelPolicy(
    role="reasoning",
    priority=40,
    keep_alive_seconds=300,
    can_evict=True,
    preload_enabled=True,
)

CODER: Final[ModelPolicy] = ModelPolicy(
    role="coder",
    priority=50,
    keep_alive_seconds=180,
    can_evict=True,
    preload_enabled=True,
)

VISION: Final[ModelPolicy] = ModelPolicy(
    role="vision",
    priority=50,
    keep_alive_seconds=180,
    can_evict=True,
    preload_enabled=True,
)

