from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from orchestrator.logging import get_logger

from .models import StreamEvent, StreamKind

logger = get_logger(__name__)


@dataclass(slots=True)
class RequestEventStream:
    request_id: str
    conversation_id: str | None = None
    max_events: int = 4096
    _events: list[StreamEvent] = field(default_factory=list)
    _seq: int = 0
    _closed: bool = False
    _closed_at: float | None = None
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def closed(self) -> bool:
        return self._closed

    async def publish(self, kind: StreamKind, **payload: Any) -> StreamEvent:
        logger.debug("STREAM: publish kind=%s payload_keys=%s", kind, sorted(payload))

        async with self._cond:
            self._seq += 1
            event = StreamEvent(
                seq=self._seq,
                kind=kind,
                request_id=self.request_id,
                conversation_id=self.conversation_id,
                payload=payload,
            )
            self._events.append(event)
            overflow = len(self._events) - max(1, self.max_events)
            if overflow > 0:
                del self._events[:overflow]
            self._cond.notify_all()
            return event

    async def subscribe(self, after_seq: int = 0) -> AsyncIterator[StreamEvent]:
        last_seq = after_seq

        while True:
            async with self._cond:
                pending = [event for event in self._events if event.seq > last_seq]
                closed = self._closed
                if not pending and not closed:
                    while not self._closed and not any(event.seq > last_seq for event in self._events):
                        await self._cond.wait()
                    pending = [event for event in self._events if event.seq > last_seq]
                    closed = self._closed

            for event in pending:
                last_seq = event.seq
                yield event

            if closed and not pending:
                break

    async def close(self) -> None:
        logger.debug("STREAM: close")
        async with self._cond:
            self._closed = True
            self._closed_at = time.monotonic()
            self._cond.notify_all()


class StreamHub:
    def __init__(self, *, max_events: int = 4096, closed_retention_s: float = 60.0) -> None:
        self._streams: dict[str, RequestEventStream] = {}
        self._max_events = max(1, max_events)
        self._closed_retention_s = max(1.0, closed_retention_s)

    def _cleanup_stale(self) -> None:
        now = time.monotonic()
        stale = [
            key
            for key, stream in self._streams.items()
            if stream.closed
            and stream._closed_at is not None
            and now - stream._closed_at >= self._closed_retention_s
        ]
        for key in stale:
            self._streams.pop(key, None)

    def get_or_create(
        self,
        request_id: str,
        conversation_id: str | None = None,
    ) -> RequestEventStream:
        self._cleanup_stale()
        stream = self._streams.get(request_id)
        if stream is None:
            stream = RequestEventStream(
                request_id=request_id,
                conversation_id=conversation_id,
                max_events=self._max_events,
            )
            self._streams[request_id] = stream
        elif conversation_id and not stream.conversation_id:
            stream.conversation_id = conversation_id
        return stream

    async def get(self, request_id: str) -> RequestEventStream | None:
        self._cleanup_stale()
        return self._streams.get(request_id)

    async def close(self, request_id: str) -> None:
        stream = self._streams.get(request_id)
        if stream is None:
            return
        await stream.close()

    async def cleanup(self) -> None:
        self._cleanup_stale()

    async def remove(self, request_id: str) -> None:
        stream = self._streams.pop(request_id, None)
        if stream is not None:
            await stream.close()
