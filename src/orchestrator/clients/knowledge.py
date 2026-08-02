from __future__ import annotations

from typing import Any

import httpx

from ..models.knowledge import KnowledgeRetrieveRequest, KnowledgeRetrieveResponse
from ..settings import Settings
from ..serialization import sanitize_for_json, validate_json_serializable, SerializationError


class KnowledgeClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        candidate_limit: int | None = None,
        neighbor_window: int | None = None,
    ) -> KnowledgeRetrieveResponse:
        request = KnowledgeRetrieveRequest(
            question=question,
            top_k=top_k if top_k is not None else self.settings.knowledge_top_k,
            candidate_limit=(
                candidate_limit
                if candidate_limit is not None
                else self.settings.knowledge_candidate_limit
            ),
            neighbor_window=(
                neighbor_window
                if neighbor_window is not None
                else self.settings.knowledge_neighbor_window
            ),
        )

        timeout = httpx.Timeout(self.settings.request_timeout_s)
        close_client = False
        client = self.client
        if client is None:
            client = httpx.AsyncClient(base_url=self.settings.knowledge_service_url, timeout=timeout)
            close_client = True

        try:
            payload = request.model_dump(exclude_none=True) if hasattr(request, "model_dump") else dict(request)
            try:
                sanitized = sanitize_for_json(payload)
                validate_json_serializable(sanitized)
            except SerializationError:
                raise
            resp = await client.post("/retrieve", json=sanitized)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return KnowledgeRetrieveResponse.model_validate(data)
        finally:
            if close_client:
                await client.aclose()

    async def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> KnowledgeRetrieveResponse:
        return await self.retrieve(question=query, top_k=top_k)


__all__ = ["KnowledgeClient"]
