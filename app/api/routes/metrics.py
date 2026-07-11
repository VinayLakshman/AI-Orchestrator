from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(tags=['metrics'])

@router.get('/metrics')
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
