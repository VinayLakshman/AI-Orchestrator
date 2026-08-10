from __future__ import annotations

from dataclasses import dataclass

from ..settings import Settings


@dataclass(slots=True)
class ModelProvider:
    """Resolve logical orchestrator roles to router model identifiers."""

    settings: Settings

    def model_for_role(self, role: str) -> str:
        role = role.lower().strip()
        mapping = {
            "controller": self.settings.controller_model_name,
            "reasoning": self.settings.reasoning_model_name,
            "coder": self.settings.coder_model_name,
            "vision": self.settings.vision_model_name,
            "embedding": self.settings.embedding_model,
        }
        if role not in mapping:
            raise KeyError(f"Unknown model role: {role}")
        return mapping[role]

    @property
    def router_base_url(self) -> str:
        return self.settings.model_router_url.rstrip("/")

    @property
    def router_origin_url(self) -> str:
        base = self.router_base_url
        if base.endswith("/v1"):
            return base[:-3]
        return base
