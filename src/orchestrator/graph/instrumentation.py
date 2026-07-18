from __future__ import annotations

from functools import wraps
from time import perf_counter
from typing import Awaitable, Callable, TypeVar

from ..models.state import OrchestratorState


StateT = TypeVar("StateT", bound=OrchestratorState)


def timed_node(
    timing_key: str,
    node: Callable[[StateT], Awaitable[StateT]],
    *,
    display_name: str | None = None,
) -> Callable[[StateT], Awaitable[StateT]]:
    label = (display_name or timing_key).strip() or timing_key

    @wraps(node)
    async def wrapper(state: StateT) -> StateT:
        started = perf_counter()
        result: StateT | None = None
        try:
            result = await node(state)
        finally:
            elapsed_ms = (perf_counter() - started) * 1000.0
            target = result if result is not None else state
            target.debug.timings[timing_key] = elapsed_ms
            target.debug.execution_trace.append(
                {
                    "key": timing_key,
                    "label": label,
                    "duration_ms": elapsed_ms,
                }
            )
        return result

    return wrapper
