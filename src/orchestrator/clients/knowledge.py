from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from ..models.knowledge import KnowledgeRetrieveRequest, KnowledgeRetrieveResponse
from ..settings import Settings


class KnowledgeServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class KnowledgeClient:
    """HTTP boundary to the independent Git indexing and retrieval service."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    def _headers(self) -> dict[str, str]:
        if self.settings.knowledge_service_api_key:
            return {"Authorization": f"Bearer {self.settings.knowledge_service_api_key}"}
        return {}

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        timeout = httpx.Timeout(self.settings.knowledge_timeout_s)
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(
            base_url=self.settings.knowledge_service_url.rstrip("/"),
            timeout=timeout,
            headers=self._headers(),
        )
        try:
            response = await client.request(method, path, headers=self._headers(), **kwargs)
            if response.is_error:
                # Do not relay upstream bodies: they may contain internal details.
                raise KnowledgeServiceError(
                    f"Knowledge Service returned HTTP {response.status_code}",
                    status_code=502 if response.status_code >= 500 else response.status_code,
                )
            if response.status_code == 204:
                return None
            return response.json()
        except httpx.TimeoutException as exc:
            raise KnowledgeServiceError("Knowledge Service request timed out", status_code=504) from exc
        except httpx.RequestError as exc:
            raise KnowledgeServiceError("Knowledge Service is unavailable", status_code=502) from exc
        except ValueError as exc:
            raise KnowledgeServiceError("Knowledge Service returned invalid JSON") from exc
        finally:
            if owns_client:
                await client.aclose()

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        candidate_limit: int | None = None,
        neighbor_window: int | None = None,
        repository: str | None = None,
        branch: str | None = None,
    ) -> KnowledgeRetrieveResponse:
        if repository or branch:
            # Do not silently ignore isolation filters on the legacy backend.
            # Enable these only after the Knowledge Service implements the
            # versioned filtered-retrieval contract.
            raise KnowledgeServiceError(
                "Repository and branch filters require the Knowledge Service v1 contract",
                status_code=501,
            )
        request = KnowledgeRetrieveRequest(
            question=question,
            top_k=top_k if top_k is not None else self.settings.knowledge_top_k,
            candidate_limit=(candidate_limit if candidate_limit is not None
                             else self.settings.knowledge_candidate_limit),
            neighbor_window=(neighbor_window if neighbor_window is not None
                             else self.settings.knowledge_neighbor_window),
            repository=repository,
            branch=branch,
        )
        # Keep the deployed legacy endpoint as retrieval fallback. Repository
        # filters only take effect once supported by the independent service.
        payload = request.model_dump(exclude_none=True)
        result = await self._request("POST", "/retrieve", json=payload)
        try:
            return KnowledgeRetrieveResponse.model_validate(result)
        except Exception as exc:
            raise KnowledgeServiceError("Knowledge Service response did not match the retrieval contract") from exc

    async def search(self, query: str, **filters: Any) -> KnowledgeRetrieveResponse:
        return await self.retrieve(question=query, **filters)

    async def list_repositories(self) -> Any:
        return await self._request("GET", "/api/v1/repositories")

    async def register_repository(self, payload: dict[str, Any]) -> Any:
        return await self._request("POST", "/api/v1/repositories", json=payload)

    async def repository_status(self, repository_id: str) -> Any:
        return await self._request("GET", f"/api/v1/repositories/{quote(repository_id, safe='')}")

    async def sync_repository(self, repository_id: str, *, full: bool = False) -> Any:
        action = "reindex" if full else "sync"
        return await self._request(
            "POST", f"/api/v1/repositories/{quote(repository_id, safe='')}/{action}", json={}
        )

    async def job_status(self, job_id: str) -> Any:
        return await self._request("GET", f"/api/v1/jobs/{quote(job_id, safe='')}")

    async def delete_repository(self, repository_id: str) -> Any:
        return await self._request(
            "DELETE", f"/api/v1/repositories/{quote(repository_id, safe='')}"
        )

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()


__all__ = ["KnowledgeClient", "KnowledgeServiceError"]
