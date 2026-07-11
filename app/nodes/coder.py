import asyncio

from app.clients.ollama import OllamaClient
from app.graph.state import OrchestratorState
from app.settings import get_settings


def coder_node(state: OrchestratorState) -> OrchestratorState:
    if not state.get('needs_coder'):
        return state
    settings = get_settings()
    client = OllamaClient(settings.ollama_base_url, timeout=settings.request_timeout_seconds)
    prompt = '\n'.join([
        'You are an internal coding assistant.',
        'Write the minimal correct solution.',
        f'User request: {state.get("user_text", "")}',
        f'Retrieved context:\n{state.get("retrieved_context", "")}',
        f'Vision notes:\n{state.get("vision_notes", "")}',
        f'Tool results:\n{state.get("tool_results", "")}',
    ])
    messages = [{'role': 'user', 'content': prompt}]
    output = asyncio.run(client.chat(settings.ollama_coder_model, messages))
    state['coder_output'] = output
    return state
