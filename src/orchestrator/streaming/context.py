from __future__ import annotations

from contextvars import ContextVar
from contextlib import asynccontextmanager
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .publisher import StreamPublisher


_current_stream: ContextVar[StreamPublisher | None] = ContextVar(
    "current_stream",
    default=None,
)


def get_current_stream() -> StreamPublisher | None:
    return _current_stream.get()


@asynccontextmanager
async def stream_scope(stream: StreamPublisher) -> AsyncIterator[StreamPublisher]:
    token = _current_stream.set(stream)
    try:
        yield stream
    finally:
        _current_stream.reset(token)