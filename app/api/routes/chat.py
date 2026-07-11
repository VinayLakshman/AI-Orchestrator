import asyncio
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.adapters.openai import build_chat_completion
from app.adapters.openwebui import extract_text_and_images
from app.core.telemetry import REQUEST_LATENCY, REQUESTS_TOTAL
from app.graph.graph import graph
from app.models.chat import OpenAIChatRequest
from app.settings import get_settings

router = APIRouter(prefix='/v1', tags=['chat'])


@router.get('/models')
async def models():
    settings = get_settings()
    return {
        'object': 'list',
        'data': [
            {'id': settings.ollama_main_model, 'object': 'model', 'owned_by': 'local'},
            {'id': settings.ollama_coder_model, 'object': 'model', 'owned_by': 'local'},
            {'id': settings.ollama_vision_model, 'object': 'model', 'owned_by': 'local'},
        ],
    }


async def _run_graph(request: OpenAIChatRequest) -> str:
    text, images = extract_text_and_images(request.messages)
    state = {
        'messages': [m.model_dump() for m in request.messages],
        'user_text': text,
        'image_urls': images,
    }
    result = await asyncio.to_thread(graph.invoke, state)
    return result.get('final_answer', '')


@router.post('/chat/completions')
async def chat_completions(payload: OpenAIChatRequest, request: Request):
    REQUESTS_TOTAL.inc()
    start = time.perf_counter()
    content = await _run_graph(payload)
    REQUEST_LATENCY.observe(time.perf_counter() - start)
    response = build_chat_completion(payload.model, content)
    if payload.stream:
        async def gen():
            chunk = {
                'id': response.id,
                'object': 'chat.completion.chunk',
                'created': response.created,
                'model': response.model,
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant', 'content': content},
                    'finish_reason': None,
                }],
            }
            yield f'data: {json.dumps(chunk)}\n\n'
            yield 'data: [DONE]\n\n'

        return StreamingResponse(gen(), media_type='text/event-stream')
    return JSONResponse(response.model_dump())
