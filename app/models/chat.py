from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str
    content: object

class OpenAIChatRequest(BaseModel):
    model: str = Field(default='qwen3:14b')
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None

class RetrievedChunk(BaseModel):
    source: str | None = None
    content: str
    score: float | None = None
