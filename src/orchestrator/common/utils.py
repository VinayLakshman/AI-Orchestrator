import json
import re
from typing import Any

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract the first JSON object from an LLM response.

    Handles:
    - plain JSON
    - markdown code fences
    - surrounding prose
    """

    if not text:
        return {}

    text = text.strip()

    # Remove markdown fences
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
