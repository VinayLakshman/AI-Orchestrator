from __future__ import annotations

import json
from typing import Any


def openai_chunk(*, id: str, model: str, content: str, request_id: str | None = None) -> str:
    payload = {
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "request_id": request_id,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def openai_done() -> str:
    return "data: [DONE]\n\n"