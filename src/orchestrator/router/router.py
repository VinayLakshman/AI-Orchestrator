from __future__ import annotations

from ..schemas import RouteDecision, RouteType
from ..settings import Settings
from .classifier import RoutingClassifier
from .detector import deterministic_route


class RequestRouter:
    def __init__(self, settings: Settings, classifier: RoutingClassifier) -> None:
        self.settings = settings
        self.classifier = classifier

    async def route(self, user_text: str) -> RouteDecision:
        deterministic = deterministic_route(user_text)
        if deterministic is not None:
            return deterministic

        classified = await self.classifier.classify(user_text)

        if classified.confidence < self.settings.routing_confidence_threshold:
            return RouteDecision(
                route=RouteType.MULTI_STEP if classified.needs_planning else RouteType.CLARIFY,
                confidence=classified.confidence,
                reason=f"Low routing confidence: {classified.reason}",
                needs_vision=classified.needs_vision,
                needs_rag=classified.needs_rag,
                needs_tools=classified.needs_tools,
                needs_code=classified.needs_code,
                needs_planning=classified.needs_planning,
                candidate_models=classified.candidate_models,
            )

        return classified

    @staticmethod
    def extract_last_user_message(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content"):
                return message["content"]
        return ""