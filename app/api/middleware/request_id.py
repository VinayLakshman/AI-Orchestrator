import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get('x-request-id', uuid.uuid4().hex)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers['x-request-id'] = request_id
        return response
