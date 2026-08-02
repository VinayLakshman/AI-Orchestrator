from __future__ import annotations

import base64
import dataclasses
import json
from collections.abc import Mapping
from dataclasses import is_dataclass, asdict
from decimal import Decimal
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Iterable

from pydantic import BaseModel


class SerializationError(TypeError):
    def __init__(self, message: str, *, path: list[str] | None = None, value: Any = None) -> None:
        super().__init__(message)
        self.path = path or []
        self.value = value


def _is_pydantic(obj: Any) -> bool:
    return isinstance(obj, BaseModel)


def canonicalize_metadata(value: Any) -> dict[str, Any]:
    """
    Ensure metadata is a plain dict[str, Any]. Convert MappingProxyType, Mapping,
    BaseModel, and dataclasses into dicts recursively.
    """
    if value is None:
        return {}
    if isinstance(value, MappingProxyType):
        value = dict(value)
    if isinstance(value, Mapping):
        return {str(k): canonicalize_metadata(v) if isinstance(v, (Mapping, BaseModel)) or is_dataclass(v) else v for k, v in dict(value).items()}
    if _is_pydantic(value):
        return canonicalize_metadata(value.model_dump(exclude_none=True))
    if is_dataclass(value):
        return canonicalize_metadata(asdict(value))
    # If it's already a dict-like python object, coerce to dict
    if isinstance(value, dict):
        return {str(k): canonicalize_metadata(v) if isinstance(v, (Mapping, BaseModel)) or is_dataclass(v) else v for k, v in value.items()}
    # Fallback: wrap in dict under 'value'
    return {"value": value}


def sanitize_for_json(obj: Any, *, _path: list[str] | None = None) -> Any:
    """
    Recursively convert Python objects into JSON-compatible primitives.
    Returns only dict, list, str, int, float, bool or None.
    Raises SerializationError with path information for unsupported objects.
    """
    path = _path or []

    # Primitives
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj

    # Decimal -> string to preserve precision
    if isinstance(obj, Decimal):
        return str(obj)

    # datetime/date -> ISO
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # MappingProxyType or Mapping
    if isinstance(obj, MappingProxyType) or isinstance(obj, Mapping):
        result: dict[str, Any] = {}
        for k, v in dict(obj).items():
            key = str(k)
            result[key] = sanitize_for_json(v, _path=path + [key])
        return result

    # dataclass
    if is_dataclass(obj):
        return sanitize_for_json(asdict(obj), _path=path)

    # Pydantic BaseModel
    if _is_pydantic(obj):
        return sanitize_for_json(obj.model_dump(exclude_none=True), _path=path)

    # Iterable types
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v, _path=path + [str(i)]) for i, v in enumerate(obj)]
    if isinstance(obj, (set, frozenset)):
        return [sanitize_for_json(v, _path=path + [str(i)]) for i, v in enumerate(sorted(obj, key=lambda x: str(x)))]

    # bytes / memoryview -> base64 string
    if isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        return base64.b64encode(b).decode("ascii")

    # Enum-like: try to extract value
    try:
        from enum import Enum

        if isinstance(obj, Enum):
            return sanitize_for_json(obj.value, _path=path)
    except Exception:
        pass

    # UUID and Path-like
    try:
        import uuid
        from pathlib import Path

        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    # numpy scalars
    try:
        import numpy as _np

        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass

    # Fallback: if object has __dict__ or is dataclass-like, try to convert
    if hasattr(obj, "__dict__"):
        try:
            return sanitize_for_json(vars(obj), _path=path)
        except Exception:
            pass

    raise SerializationError(f"Type not JSON serializable: {type(obj)!r}", path=path, value=obj)


def validate_json_serializable(obj: Any) -> None:
    """Sanitize and validate that `obj` can be encoded by json.dumps."""
    sanitized = sanitize_for_json(obj)
    try:
        json.dumps(sanitized, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - surface symptoms only
        # Wrap into our SerializationError if needed
        raise SerializationError(f"json.dumps failed: {exc}", path=[], value=obj)


__all__ = [
    "sanitize_for_json",
    "validate_json_serializable",
    "SerializationError",
    "canonicalize_metadata",
]
