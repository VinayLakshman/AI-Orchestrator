import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

class KnowledgeServiceClient:
    def __init__(self, base_url: str, retrieve_path: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.retrieve_path = retrieve_path
        self.timeout = timeout

    async def retrieve(self, query: str, top_k: int = 8, repositories: list[str] | None = None) -> list[dict[str, Any]]:
        payload = {'query': query, 'top_k': top_k, 'repositories': repositories or []}
        url = f'{self.base_url}{self.retrieve_path}'
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
        except Exception as exc:
            logger.warning('knowledge retrieval failed: %s', exc)
            return []
        if isinstance(data, dict):
            for key in ('chunks', 'results', 'documents', 'items'):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        if isinstance(data, list):
            return data
        return []
