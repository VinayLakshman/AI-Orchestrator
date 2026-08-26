from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..runtime.model_provider import ModelProvider
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
    - inference client

    The router owns model residency; this manager only exposes logical role
    metadata and the shared inference client registry.
    """

    def __init__(
        self,
        settings: Settings,
        client_registry: Any,
        provider: ModelProvider | None = None,
    ) -> None:
        self.settings = settings
        self.client_registry = client_registry
        self.provider = provider or ModelProvider(settings)

    def controller(self) -> ManagedModel:
        return ManagedModel("controller", self.provider.model_for_role("controller"))

    def reasoning(self) -> ManagedModel:
        return ManagedModel("reasoning", self.provider.model_for_role("reasoning"))

    def coder(self) -> ManagedModel:
        return ManagedModel("coder", self.provider.model_for_role("coder"))

    def vision(self) -> ManagedModel:
        return ManagedModel("vision", self.provider.model_for_role("vision"))

    def embedding(self) -> ManagedModel:
        return ManagedModel("embedding", self.settings.embedding_model)

    def model_for_role(self, role: str) -> str:
        return self.provider.model_for_role(role)

    def container_for_role(self, role: str) -> str:
        """Return the container name for a given model role.
        
        Used by ModelLifecycle for GPU ownership management.
        """
        model_name = self.provider.model_for_role(role)
        # Default to container name being the same as model name for llama-router containers
        return model_name

    def client(self, role: str) -> Any:
        """Return the inference client for a model role."""
        return self.client_registry.get(role)
