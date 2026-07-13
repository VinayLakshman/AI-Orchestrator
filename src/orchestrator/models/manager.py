from __future__ import annotations

from dataclasses import dataclass

from ..clients.ollama import OllamaClient
from ..settings import Settings
from ..models.chat import ChatRole, ChatMessage


@dataclass(slots=True)
class ManagedModel:
    role: str
    name: str


class ModelManager:
    """
    Role-based model registry.

    The orchestrator references model roles, never raw model names.
    """

    def __init__(self, settings: Settings, ollama_client: OllamaClient) -> None:
        self.settings = settings
        self.ollama_client = ollama_client

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

    async def warm_controller(self) -> None:
        """
        Load the controller into VRAM and keep it warm.
        """
        await self.ollama_client.chat(
            model=self.settings.controller_model,
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="You are a resident controller. Reply with OK."),
                ChatMessage(role=ChatRole.USER, content="Warm up and stay resident."),
            ],
            temperature=0.0,
            max_tokens=4,
            stream=False,
            keep_alive=self.settings.controller_keep_alive,
        )