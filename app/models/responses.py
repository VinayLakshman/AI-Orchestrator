from pydantic import BaseModel

class ChatResponseMessage(BaseModel):
    role: str = 'assistant'
    content: str

class ChatChoice(BaseModel):
    index: int = 0
    message: ChatResponseMessage
    finish_reason: str = 'stop'

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = 'chat.completion'
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage = Usage()
