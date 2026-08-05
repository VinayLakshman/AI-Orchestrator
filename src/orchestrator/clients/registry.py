from __future__ import annotations

from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)


class ClientRegistry:
    """Registry mapping model role -> inference client.

    The registry is the single place callers obtain a client for a given
    model role. It is intentionally backend-agnostic so the orchestrator
    never needs to know which inference engine is in use.
    """

    def __init__(self, clients: dict[str, Any] | None = None) -> None:
        self._clients: dict[str, Any] = dict(clients or {})

    def register(self, role: str, client: Any) -> None:
        role = role.lower().strip()
        self._clients[role] = client

    def get(self, role: str) -> Any:
        role = role.lower().strip()
        if role not in self._clients:
            raise KeyError(f"No inference client registered for model role: {role}")
        return self._clients[role]

    def has(self, role: str) -> bool:
        return role.lower().strip() in self._clients

    def roles(self) -> list[str]:
        return list(self._clients.keys())

    def clients(self) -> list[Any]:
        return list(self._clients.values())
