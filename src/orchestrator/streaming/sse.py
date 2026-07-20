from __future__ import annotations

import json
from time import time
from typing import Any


def openai_chunk(
    *,
    request_id: str,
    model: str,
    role: str | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
) -> str:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content

    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def openai_done() -> str:
    return "data: [DONE]\n\n"