# AI Orchestrator

AI Orchestrator is a FastAPI service that exposes an OpenAI-compatible chat API and routes each request through a fixed LangGraph workflow. It is intended to be the single inference entry point for a homelab AI stack backed by Ollama, a Knowledge Service, and an optional MCP Gateway.

This repository owns the application code and Docker image build. Runtime composition is expected to live outside this repository.

## Current State

Implemented:

- `GET /health` liveness endpoint.
- `GET /metrics` Prometheus metrics endpoint.
- `GET /v1/models` OpenAI-style model listing based on configured local model names.
- `POST /v1/chat/completions` OpenAI-compatible chat completion endpoint.
- LangGraph workflow with these nodes: `classify`, `retrieve`, `vision`, `tools`, `coder`, `synthesize`, `finish`.
- Ollama chat adapter for main, coder, and vision model calls.
- Knowledge Service retrieval client using `POST /retrieve`.
- MCP Gateway client stub using `POST /tools/execute`.
- Open WebUI/OpenAI-style text and image content extraction.
- Optional SSE response mode for chat completions.
- Request ID middleware using incoming `x-request-id` or a generated ID.
- Config via environment variables and `.env`.
- Dockerfile-based container build.
- Unit and integration tests for current API and adapter behavior.

Accepted but not fully implemented:

- `temperature` and `max_tokens` are accepted on chat completion requests for compatibility, but they are not currently forwarded to Ollama.
- Streaming begins with an assistant role-only chunk, then emits assistant content chunks followed by `[DONE]`.
- The `model` request field is echoed in the response. Internal graph nodes use configured models from environment variables.

Not implemented in this repository:

- A repository-owned Docker Compose deployment.
- Authentication or authorization.
- Persistent storage.
- True multi-agent planning.
- Full MCP tool registry or local tool execution framework.
- Dependency-aware readiness endpoint.

## How Requests Work

The service receives an OpenAI-compatible chat completion request at `/v1/chat/completions`.

Only `user` messages are extracted into graph input:

- String content is appended to `user_text`.
- Content parts with `{"type": "text"}` are appended to `user_text`.
- Content parts with `{"type": "image_url"}` are collected as `image_urls`.

The graph always executes the same node order:

```text
START
  -> classify
  -> retrieve
  -> vision
  -> tools
  -> coder
  -> synthesize
  -> finish
END
```

Nodes are internally gated by classification flags:

| Node | What it does |
| --- | --- |
| `classify` | Sets `needs_retrieval`, `needs_vision`, `needs_coder`, and `needs_tools`. Routing is decided by the Controller LLM, which produces an `ExecutionPlan` / execution queue; the normalizer only performs structural normalization (attachments, URLs, code blocks) — it never routes on keywords. |
| `retrieve` | Calls the Knowledge Service when `needs_retrieval=true`; formats returned chunks into `retrieved_context`. |
| `vision` | Calls the configured Ollama vision model when images are present or vision is requested. |
| `tools` | Calls the MCP Gateway stub with tool name `inspect` when tool use is requested. |
| `coder` | Calls the configured Ollama coder model when code-related work is requested. |
| `synthesize` | Calls the configured main Ollama model to produce the final answer from the user request plus any intermediate context. |
| `finish` | Ensures `final_answer` exists, falling back to intermediate outputs or `No answer generated.` |

Classification flags are decided by the Controller LLM when producing the
`ExecutionPlan`; there is no deterministic keyword table.

## Runtime Dependencies

Expected external services:

- Ollama at `OLLAMA_URL`.
- Knowledge Service at `KNOWLEDGE_SERVICE_URL`.
- Optional MCP Gateway at `MCP_URL`.

The service degrades differently by dependency:

- Knowledge Service failures return no retrieved chunks and continue.
- MCP Gateway failures are captured as tool-result errors and continue.
- Ollama failures return empty model output; the graph then relies on fallback behavior in later nodes.

## Configuration

Configuration is loaded from environment variables and `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_NAME` | `ai-orchestrator` | Application name. |
| `APP_ENV` | `development` | Runtime environment label. |
| `APP_HOST` | `0.0.0.0` | Host value for external launch scripts. |
| `APP_PORT` | `8000` | Port value for external launch scripts. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama base URL. |
| `LLM_MAIN_MODEL` | `qwen3:14b` | Main synthesis model. |
| `LLM_CODER_MODEL` | `qwen2.5-coder:7b` | Coder node model. |
| `LLM_VISION_MODEL` | `qwen2.5-vl:7b` | Vision node model. |
| `KNOWLEDGE_SERVICE_URL` | `http://knowledge-service:8001` | Knowledge Service base URL. |
| `MCP_URL` | `http://mcp-gateway:9000` | MCP Gateway base URL. |
| `REQUEST_TIMEOUT_SECONDS` | `60.0` | HTTP client timeout for external service calls. |
| `MAX_CONTEXT_CHUNKS` | `8` | Retrieval `top_k` sent to Knowledge Service. |
| `ENABLE_STREAMING` | `true` | Enables SSE responses when request `stream=true`. |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated CORS origins. |

## Build And Run

Build the container image:

```bash
docker build -f docker/Dockerfile -t ai-orchestrator .
```

Run the container directly:

```bash
cp .env.example .env
docker run --env-file .env -p 8000:8000 ai-orchestrator
```

Run locally for development:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run tests:

```bash
.venv/bin/pytest
```

Run the syntax check used by `scripts/lint.sh`:

```bash
python3 -m compileall app
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check. |
| `GET` | `/metrics` | Prometheus metrics. |
| `GET` | `/v1/models` | Configured local model IDs. |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions. |

## `GET /health`

Returns:

```json
{"status": "ok"}
```

## `GET /metrics`

Returns Prometheus metrics including:

- `orchestrator_requests_total`
- `orchestrator_request_latency_seconds`

## `GET /v1/models`

Example request:

```bash
curl http://localhost:8000/v1/models
```

Example response:

```json
{
  "object": "list",
  "data": [
    {"id": "qwen3:14b", "object": "model", "owned_by": "local"},
    {"id": "qwen2.5-coder:7b", "object": "model", "owned_by": "local"},
    {"id": "qwen2.5-vl:7b", "object": "model", "owned_by": "local"}
  ]
}
```

## `POST /v1/chat/completions`

Supported request fields:

| Field | Type | Required | Default | Behavior |
| --- | --- | --- | --- | --- |
| `model` | string | no | `qwen3:14b` | Echoed in the response. Does not override internal node model selection. |
| `messages` | array | yes | none | Chat messages. Only `user` messages are extracted into graph input. |
| `messages[].role` | string | yes | none | Common values are `system`, `user`, and `assistant`; only `user` affects orchestration. |
| `messages[].content` | string or array | yes | none | Plain text or content parts. |
| `stream` | boolean | no | `false` | Returns SSE when `true` and `ENABLE_STREAMING=true`. |
| `temperature` | number or null | no | `null` | Accepted but not forwarded to Ollama. |
| `max_tokens` | integer or null | no | `null` | Accepted but not forwarded to Ollama. |

Supported content parts:

| Part | Shape | Behavior |
| --- | --- | --- |
| Text | `{"type": "text", "text": "..."}` | Appended to `user_text`. |
| Image URL | `{"type": "image_url", "image_url": {"url": "..."}}` | Appended to `image_urls`; enables vision handling. |

Image URLs may be remote URLs or `data:` URLs. Data URLs are normalized before being sent to Ollama.

Minimal text request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:14b",
    "messages": [
      {"role": "user", "content": "Explain the homelab orchestrator."}
    ]
  }'
```

Text and image request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:14b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is shown here?"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
      }
    ]
  }'
```

Streaming request:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:14b",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Give me a short status summary."}
    ]
  }'
```

Non-streaming response shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "qwen3:14b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Streaming response shape (the role-only chunk is sent immediately; content chunks follow):

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1760000000,"model":"qwen3:14b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1760000000,"model":"qwen3:14b","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}

: keep-alive

data: [DONE]
```

The `: keep-alive` line is an SSE comment emitted every 10 seconds while orchestration is running and no assistant token has arrived. It is not an OpenAI event or conversation content.

## Repository Map

```text
app/
  main.py                 FastAPI application setup and mounted routes.
  settings.py             Environment-backed settings.
  api/
    routes/               Health, metrics, and OpenAI-compatible chat routes.
    middleware/           Request ID middleware.
  adapters/               OpenAI response builder and Open WebUI content parser.
  clients/                HTTP clients for Ollama, Knowledge Service, and MCP Gateway.
  graph/                  LangGraph builder, state type, and specialist nodes.
  nodes/                  Graph node implementations.
  models/                 Pydantic request/response models.
  tools/                  Placeholder module boundary for future MCP-backed helpers.
config/                   Static YAML config examples; runtime settings currently come from env/.env.
docker/                   Dockerfile and entrypoint.
docs/                     Placeholder documentation files.
scripts/                  Small local helper scripts.
tests/                    Unit and integration tests.
```

## Known Limitations

- Routing is planned by the Controller LLM (semantic), not deterministic.
- The graph executes every node in fixed order; nodes no-op when their classification flag is false.
- Streaming is response-shape compatible but not token incremental.
- Token usage values in responses are currently zero.
- Chat request `temperature` and `max_tokens` are compatibility fields only.
- There is no repository-local deployment composition by design.
- The docs under `docs/` are placeholders; the root README is the current source of project orientation.
