"""Controller response parsing utilities."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from a controller response."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = _JSON_RE.search(text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


__all__ = ["extract_json_object"]
