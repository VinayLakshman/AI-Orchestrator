from __future__ import annotations

from typing import Any, Iterable

from ..common.enums import SpecialistType


def normalize_specialist(value: Any) -> SpecialistType | None:
    if value is None:
        return None

    if isinstance(value, SpecialistType):
        return value

    try:
        return SpecialistType(str(value).strip().lower())
    except Exception:
        return None


def unique_specialists(values: Iterable[Any]) -> list[SpecialistType]:
    seen: set[SpecialistType] = set()
    result: list[SpecialistType] = []

    for value in values:
        specialist = normalize_specialist(value)

        if specialist is None:
            continue

        if specialist in seen:
            continue

        seen.add(specialist)
        result.append(specialist)

    return result


def unique_specialist_values(values: Iterable[Any]) -> list[str]:
    return [step.value for step in unique_specialists(values)]