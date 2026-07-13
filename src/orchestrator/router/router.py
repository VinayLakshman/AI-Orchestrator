from __future__ import annotations

from dataclasses import dataclass

from ..schemas import RouteDecision
from .classifier import RoutingClassifier


@dataclass(slots=True)
class RequestRouter:
    classifier: RoutingClassifier

    def route(self, text: str) -> RouteDecision:
        return self.classifier.classify(text)
