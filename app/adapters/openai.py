import time
import uuid
from app.models.responses import ChatChoice, ChatCompletionResponse, ChatResponseMessage


def build_chat_completion(model: str, content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f'chatcmpl-{uuid.uuid4().hex[:24]}',
        created=int(time.time()),
        model=model,
        choices=[ChatChoice(message=ChatResponseMessage(content=content))],
    )
