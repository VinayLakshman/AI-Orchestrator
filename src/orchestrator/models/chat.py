from typing import Any

from pydantic import BaseModel, Field

from ..common.enums import ChatRole
from ..common.types import MessageContent

class ChatMessage(BaseModel):
    role: ChatRole
    content: MessageContent
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    thread_id: str | None = None
    model: str | None = None
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
