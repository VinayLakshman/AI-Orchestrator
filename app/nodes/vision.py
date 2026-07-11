import asyncio

from app.clients.ollama import OllamaClient
from app.graph.state import OrchestratorState
from app.settings import get_settings


def vision_node(state: OrchestratorState) -> OrchestratorState:
    if not state.get('needs_vision'):
        return state
    settings = get_settings()
    images = state.get('image_urls', [])
    if not images:
        return state
    client = OllamaClient(settings.ollama_base_url, timeout=settings.request_timeout_seconds)
    prompt = state.get('user_text', '') or 'Inspect the image and summarize what matters.'
    payload_messages = [{'role': 'user', 'content': prompt}]
    normalized_images = [client.normalize_data_url(img) for img in images]
    notes = asyncio.run(client.chat(settings.ollama_vision_model, payload_messages, images=normalized_images))
    state['vision_notes'] = notes
    return state
