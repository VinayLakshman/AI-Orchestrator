from app.graph.router import classify
from app.graph.state import OrchestratorState
from app.settings import get_settings


def classify_node(state: OrchestratorState) -> OrchestratorState:
    settings = get_settings()
    text = state.get('user_text', '')
    image_urls = state.get('image_urls', [])
    flags = classify(text, image_urls, settings)
    state.update(flags)
    return state
