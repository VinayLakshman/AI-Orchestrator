import httpx
import pytest
import respx

from app.clients.ollama import OllamaClient


@pytest.mark.asyncio
@respx.mock
async def test_chat_returns_message_content():
    respx.post('http://ollama.test/api/chat').mock(
        return_value=httpx.Response(200, json={'message': {'content': 'hello'}})
    )

    content = await OllamaClient('http://ollama.test').chat('model-a', [{'role': 'user', 'content': 'hi'}])

    assert content == 'hello'


@pytest.mark.asyncio
@respx.mock
async def test_chat_returns_empty_string_on_backend_error():
    respx.post('http://ollama.test/api/chat').mock(return_value=httpx.Response(503))

    content = await OllamaClient('http://ollama.test').chat('model-a', [{'role': 'user', 'content': 'hi'}])

    assert content == ''
