from fastapi.testclient import TestClient

from app.api.routes import chat as chat_route
from app.main import app


class FakeGraph:
    def invoke(self, state):
        return {'final_answer': f'echo: {state["user_text"]}'}


def test_health_includes_request_id_header():
    client = TestClient(app)

    response = client.get('/health', headers={'x-request-id': 'test-request-id'})

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
    assert response.headers['x-request-id'] == 'test-request-id'


def test_metrics_route_is_mounted():
    client = TestClient(app)

    response = client.get('/metrics')

    assert response.status_code == 200
    assert 'orchestrator_requests_total' in response.text


def test_chat_completion_returns_openai_compatible_response(monkeypatch):
    monkeypatch.setattr(chat_route, 'graph', FakeGraph())
    client = TestClient(app)

    response = client.post(
        '/v1/chat/completions',
        json={'model': 'local-model', 'messages': [{'role': 'user', 'content': 'hello'}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['object'] == 'chat.completion'
    assert body['model'] == 'local-model'
    assert body['choices'][0]['message']['content'] == 'echo: hello'
