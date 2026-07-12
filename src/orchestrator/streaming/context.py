from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar

from .publisher import StreamPublisher


_current_stream: ContextVar[StreamPublisher | None] = ContextVar(
    "current_stream",
    default=None,
)


def get_current_stream() -> StreamPublisher | None:
    return _current_stream.get()


@asynccontextmanager
async def stream_scope(publisher: StreamPublisher):
    token = _current_stream.set(publisher)
    try:
        yield publisher
    finally:
        _current_stream.reset(token)