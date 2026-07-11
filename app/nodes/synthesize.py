import asyncio

from app.clients.ollama import OllamaClient
from app.graph.state import OrchestratorState
from app.settings import get_settings


def synthesize_node(state: OrchestratorState) -> OrchestratorState:
    settings = get_settings()
    client = OllamaClient(settings.ollama_base_url, timeout=settings.request_timeout_seconds)
    system = 'You are the primary local assistant. Use retrieved context and internal outputs to answer directly.'
    prompt_parts = [
        system,
        f'User request: {state.get("user_text", "")}',
        f'Retrieved context:\n{state.get("retrieved_context", "")}',
        f'Vision notes:\n{state.get("vision_notes", "")}',
        f'Coder output:\n{state.get("coder_output", "")}',
        f'Tool results:\n{state.get("tool_results", "")}',
    ]
    messages = [{'role': 'user', 'content': '\n\n'.join([p for p in prompt_parts if p])}]
    answer = asyncio.run(client.chat(settings.ollama_main_model, messages))
    state['final_answer'] = answer
    return state
