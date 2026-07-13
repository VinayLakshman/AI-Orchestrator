from __future__ import annotations

from dataclasses import dataclass

from ..schemas import RouteDecision
from .detector import deterministic_route


@dataclass(slots=True)
class RoutingClassifier:
    def classify(self, text: str) -> RouteDecision:
        return deterministic_route(text)