"""llama-router residency status vocabulary."""

ROUTER_READY_STATUSES = frozenset({"loaded", "ready", "running", "sleeping"})
ROUTER_TRANSITIONAL_STATUSES = frozenset({"loading"})

__all__ = ["ROUTER_READY_STATUSES", "ROUTER_TRANSITIONAL_STATUSES"]
