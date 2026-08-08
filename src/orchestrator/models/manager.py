from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..settings import Settings


@dataclass(slots=True)
class ManagedModel:
    role: str
    name: str


class ModelManager:
    """
    Central model metadata registry.

    Owns the mappings from model role to:
    - model name
    - endpoint
    - container name
    - inference client

    The lifecycle manager and other components obtain model metadata from
    here rather than maintaining their own copies.
    """

    def __init__(self, settings: Settings, client_registry: Any) -> None:
        self.settings = settings
        self.client_registry = client_registry

    def controller(self) -> ManagedModel:
        return ManagedModel("controller", self.settings.controller_model)

    def reasoning(self) -> ManagedModel:
        return ManagedModel("reasoning", self.settings.reasoning_model)

    def coder(self) -> ManagedModel:
        return ManagedModel("coder", self.settings.coder_model)

    def vision(self) -> ManagedModel:
        return ManagedModel("vision", self.settings.vision_model)

    def embedding(self) -> ManagedModel:
        return ManagedModel("embedding", self.settings.embedding_model)

    def model_for_role(self, role: str) -> str:
        role = role.lower().strip()
        mapping = {
            "controller": self.settings.controller_model,
            "reasoning": self.settings.reasoning_model,
            "coder": self.settings.coder_model,
            "vision": self.settings.vision_model,
            "embedding": self.settings.embedding_model,
        }
        if role not in mapping:
            raise KeyError(f"Unknown model role: {role}")
        return mapping[role]

    def endpoint_for_role(self, role: str) -> str:
        role = role.lower().strip()
        model_config = self._model_config(role)
        return model_config.endpoint

    def container_for_role(self, role: str) -> str:
        role = role.lower().strip()
        model_config = self._model_config(role)
        return model_config.container_name

    def client(self, role: str) -> Any:
        """Return the inference client for a model role."""
        return self.client_registry.get(role)

    def _model_config(self, role: str) -> Any:
        role = role.lower().strip()
        models = getattr(self.settings, "models", {}) or {}
        if role not in models:
            raise KeyError(f"Unknown model role: {role}")
        return models[role]
