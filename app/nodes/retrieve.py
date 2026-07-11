import asyncio

from app.clients.knowledge import KnowledgeServiceClient
from app.graph.state import OrchestratorState
from app.settings import get_settings


def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ''
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get('source') or chunk.get('path') or chunk.get('file') or 'unknown'
        content = chunk.get('content') or chunk.get('text') or ''
        score = chunk.get('score')
        header = f'[{i}] {source}'
        if score is not None:
            header += f' (score={score})'
        lines.append(f'{header}\n{content}')
    return '\n\n'.join(lines)


def retrieve_node(state: OrchestratorState) -> OrchestratorState:
    if not state.get('needs_retrieval'):
        return state
    settings = get_settings()
    client = KnowledgeServiceClient(
        base_url=settings.knowledge_service_url,
        retrieve_path=settings.knowledge_retrieve_path,
        timeout=settings.request_timeout_seconds,
    )
    query = state.get('user_text', '')
    chunks = asyncio.run(client.retrieve(query=query, top_k=settings.max_context_chunks))
    state['retrieved_context'] = _build_context(chunks)
    return state
