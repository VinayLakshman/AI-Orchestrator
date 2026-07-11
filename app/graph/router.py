from app.settings import Settings

KEYWORDS_RETRIEVAL = ('what does', 'how is', 'docs', 'knowledge', 'homelab', 'where is', 'why is', 'explain')
KEYWORDS_CODE = ('fix', 'patch', 'update', 'edit', 'generate code', 'compose', 'dockerfile', 'python', 'code')
KEYWORDS_TOOLS = ('run', 'restart', 'check logs', 'logs', 'status', 'shell', 'docker', 'git', 'home assistant')
KEYWORDS_VISION = ('image', 'screenshot', 'photo', 'diagram', 'visual', 'see this')


def classify(text: str, image_urls: list[str], settings: Settings) -> dict[str, bool]:
    lower = (text or '').lower()
    needs_vision = bool(image_urls) or any(k in lower for k in KEYWORDS_VISION)
    needs_retrieval = any(k in lower for k in KEYWORDS_RETRIEVAL)
    needs_coder = any(k in lower for k in KEYWORDS_CODE)
    needs_tools = any(k in lower for k in KEYWORDS_TOOLS)
    return {
        'needs_vision': needs_vision,
        'needs_retrieval': needs_retrieval,
        'needs_coder': needs_coder,
        'needs_tools': needs_tools,
    }
