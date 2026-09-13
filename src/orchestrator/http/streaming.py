"""Graph-to-event-stream coordination for HTTP routes."""

from __future__ import annotations

from typing import Any

from ..graph.build import OrchestratorRuntime
from ..streaming.context import stream_scope
from ..streaming.publisher import StreamPublisher


async def run_graph_with_stream(
    *,
    runtime: OrchestratorRuntime,
    request_id: str,
    thread_id: str,
    state_input: Any,
    publisher: StreamPublisher,
) -> Any:
    try:
        async with stream_scope(publisher):
            await publisher.graph_started()
            result = await runtime.graph.ainvoke(
                {"request": state_input.request},
                config={"configurable": {"thread_id": thread_id}},
            )
            await publisher.graph_finished(route=result.execution.plan.route.value)
            return result
    except Exception as exc:
        await publisher.graph_failed(str(exc))
        raise
    finally:
        await runtime.stream_hub.close(request_id)


__all__ = ["run_graph_with_stream"]
