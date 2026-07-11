import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self, base_url: str = '', execute_path: str = '/tools/execute', timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.execute_path = execute_path
        self.timeout = timeout

    async def execute(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            return {'ok': False, 'error': 'mcp gateway not configured', 'tool': tool}
        payload = {'tool': tool, 'arguments': arguments}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(f'{self.base_url}{self.execute_path}', json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            logger.warning('mcp execution failed: %s', exc)
            return {'ok': False, 'error': str(exc), 'tool': tool}
