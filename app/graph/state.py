from typing import TypedDict, Any

class OrchestratorState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    user_text: str
    image_urls: list[str]
    needs_retrieval: bool
    needs_vision: bool
    needs_coder: bool
    needs_tools: bool
    retrieved_context: str
    vision_notes: str
    coder_output: str
    tool_results: str
    final_answer: str
