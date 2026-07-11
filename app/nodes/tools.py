import asyncio

from app.clients.mcp import MCPClient
from app.graph.state import OrchestratorState
from app.settings import get_settings


def tools_node(state: OrchestratorState) -> OrchestratorState:
    if not state.get('needs_tools'):
        return state
    settings = get_settings()
    client = MCPClient(settings.mcp_gateway_url, execute_path=settings.mcp_execute_path, timeout=settings.request_timeout_seconds)
    result = asyncio.run(client.execute('inspect', {'query': state.get('user_text', '')}))
    state['tool_results'] = str(result)
    return state
