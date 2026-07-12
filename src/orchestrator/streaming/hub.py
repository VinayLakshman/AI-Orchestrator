from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from time import time

from .models import StreamEvent


_SENTINEL = object()


@dataclass
class RequestEventStream:
    request_id: str
    conversation_id: str | None = None
    created_at: float = field(default_factory=time)
    ttl_seconds: int = 900

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _buffer: list[StreamEvent] = field(default_factory=list, init=False)
    _subscribers: set[asyncio.Queue] = field(default_factory=set, init=False)
    _closed: bool = field(default=False, init=False)
    _seq: int = field(default=0, init=False)
    _closed_at: float | None = field(default=None, init=False)

    async def publish(self, event: StreamEvent) -> StreamEvent:
        async with self._lock:
            if self._closed:
                return event

            self._seq += 1
            event.seq = self._seq
            self._buffer.append(event)
            subscribers = list(self._subscribers)

        for queue in subscribers:
            queue.put_nowait(event)

        return event

    async def subscribe(self, after_seq: int = 0) -> AsyncIterator[StreamEvent]:
        queue: asyncio.Queue = asyncio.Queue()

        async with self._lock:
            backlog = [e for e in self._buffer if e.seq > after_seq]
            self._subscribers.add(queue)

        try:
            for event in backlog:
                yield event

            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, StreamEvent) and item.seq > after_seq:
                    yield item
        finally:
            async with self._lock:
                self._subscribers.discard(queue)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._closed_at = time()
            subscribers = list(self._subscribers)

        for queue in subscribers:
            queue.put_nowait(_SENTINEL)

    def expired(self) -> bool:
        if self._closed_at is None:
            return False
        return (time() - self._closed_at) > self.ttl_seconds


class StreamHub:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._streams: dict[str, RequestEventStream] = {}

    async def create(self, request_id: str, conversation_id: str | None = None) -> RequestEventStream:
        async with self._lock:
            stream = RequestEventStream(
                request_id=request_id,
                conversation_id=conversation_id,
            )
            self._streams[request_id] = stream
            return stream

    async def get(self, request_id: str) -> RequestEventStream | None:
        async with self._lock:
            return self._streams.get(request_id)

    async def close(self, request_id: str) -> None:
        stream = await self.get(request_id)
        if stream:
            await stream.close()

    async def cleanup(self) -> None:
        async with self._lock:
            expired = [rid for rid, s in self._streams.items() if s.expired()]
            for rid in expired:
                self._streams.pop(rid, None)