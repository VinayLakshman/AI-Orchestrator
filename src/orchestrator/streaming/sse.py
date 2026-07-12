from __future__ import annotations

import json
import time
from typing import Any


def openai_chunk(
    *,
    request_id: str,
    model: str,
    delta: dict[str, Any],
    created: int | None = None,
    finish_reason: str | None = None,
    index: int = 0,
) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def openai_done() -> str:
    return "data: [DONE]\n\n"