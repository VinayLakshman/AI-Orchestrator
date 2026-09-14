"""Compatibility helpers retained from the original common namespace."""

from .compatibility import compatibility_exports
from ..controller.parsing import extract_json_object

_extract_json_object = extract_json_object

__all__ = compatibility_exports(
    "orchestrator.common.utils",
    ("_extract_json_object",),
)
