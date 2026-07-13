from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .models import StreamEvent, StreamKind


@dataclass
class RequestEventStream:
    request_id: str
    conversation_id: str | None = None
    _events: list[StreamEvent] = field(default_factory=list)
    _seq: int = 0
    _closed: bool = False
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def publish(self, kind: StreamKind, **payload: Any) -> StreamEvent:
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
            self._cond.notify_all()
            return event

    async def subscribe(self, after_seq: int = 0):
        idx = 0
        while True:
            async with self._cond:
                while idx >= len(self._events) and not self._closed:
                    await self._cond.wait()

                while idx < len(self._events):
                    event = self._events[idx]
                    idx += 1
                    if event.seq <= after_seq:
                        continue
                    yield event

                if self._closed:
                    break

    async def close(self) -> None:
        async with self._cond:
            self._closed = True
            self._cond.notify_all()


class StreamHub:
    def __init__(self) -> None:
        self._streams: dict[str, RequestEventStream] = {}

    def get_or_create(self, request_id: str, conversation_id: str | None = None) -> RequestEventStream:
        stream = self._streams.get(request_id)
        if stream is None:
            stream = RequestEventStream(request_id=request_id, conversation_id=conversation_id)
            self._streams[request_id] = stream
        elif conversation_id and not stream.conversation_id:
            stream.conversation_id = conversation_id
        return stream

    async def get(self, request_id: str) -> RequestEventStream | None:
        return self._streams.get(request_id)

    async def cleanup(self) -> None:
        stale = [key for key, stream in self._streams.items() if stream._closed]
        for key in stale:
            self._streams.pop(key, None)