import base64
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    async def chat(self, model: str, messages: list[dict[str, Any]], images: list[str] | None = None) -> str:
        payload = {'model': model, 'messages': messages, 'stream': False}
        if images:
            if messages:
                payload['messages'] = list(messages)
                payload['messages'][-1] = dict(payload['messages'][-1])
                payload['messages'][-1]['images'] = images
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f'{self.base_url}/api/chat', json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get('message', {}).get('content', '')

    @staticmethod
    def normalize_data_url(data_url: str) -> str:
        if data_url.startswith('data:') and ',' in data_url:
            return data_url.split(',', 1)[1]
        return data_url
