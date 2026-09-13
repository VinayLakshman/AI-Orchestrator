# AI Orchestrator

AI Orchestrator is a FastAPI/LangGraph service that presents a stable
OpenAI-compatible interface while coordinating local controller, reasoning,
coding, vision, retrieval, web-search, tools, and image-generation paths.
It is designed for a single-GPU deployment where only one model or image
workload is active at a time.

The optimized default execution mode is ChatGPT-like: requests can share a
conversation thread, multimodal evidence can be reused, simple requests use
shorter internal paths, and streaming sends the assistant role chunk before
model-backed preprocessing starts. Internal planner, validator, and lifecycle
events are never included in the assistant answer.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | Liveness response. |
| GET | `/readyz` | Runtime, graph, and checkpointer readiness. |
| GET | `/metrics` | Dependency-free Prometheus-compatible timing metrics. |
| GET | `/v1/models` | OpenAI-style local model list. |
| POST | `/chat` | Native orchestrator request/response API. |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions. |
| GET | `/v1/streams/{request_id}` | Replayable internal/event SSE stream. |

The container and default application port are `8001`.

## OpenAI-compatible conversation identity

`POST /v1/chat/completions` remains stateless when no identity is supplied.
For persistent conversation state, provide either:

- `X-Orchestrator-Thread-ID: <stable-id>`; or
- `metadata.thread_id` in the JSON request.

The same stable ID must be sent on later turns. `/chat` already supports its
native `thread_id` field. The checkpointer determines whether that state also
survives process restarts.

## Request flow

The controller produces a semantic `ExecutionPlan`; routing is not based on a
keyword table. The graph then executes only the planned specialists:

```text
prepare -> plan -> specialist queue -> validate when needed -> finalize
                                      \\-> image_generation -> END
```

Supported specialist paths are:

- Knowledge Service retrieval/RAG.
- SearXNG web search with conservative follow-up evidence reuse.
- Vision for image URLs, data URLs, and mixed text/image messages.
- Coding and reasoning model calls.
- MCP/tool planning with explicit disabled and not-executed states.
- Open WebUI image generation as a terminal route.

The graph retains the legacy validation/retry behavior for multi-specialist,
failed, uncertain, or explicitly legacy executions. A high-confidence,
single-specialist success may go directly to finalization. Query rewriting is
also deferred when the normalized user request is already a usable retrieval
query.

## GPU and model lifecycle

All controller, reasoning, coder, vision, planner, resolver, retrieval-query,
validation, and finalizer calls use the shared router client and inference
gateway. The gateway:

- serializes complete LLM calls through a fair workload queue;
- keeps the lease through the complete streaming response;
- ensures the configured router model is ready before inference;
- reuses persistent HTTP connections;
- prevents an image-generation request from racing an LLM workload.

The existing ComfyUI release barrier remains authoritative. Image generation
does not return until Open WebUI completes and GPU release has been verified by
real memory observables. An unknown image workload retains the safety barrier.

## Configuration

Settings are read from `.env` and environment variables using Pydantic
Settings. Field names map to uppercase environment variables, for example
`MODEL_ROUTER_URL` and `ENABLE_RAG`.

Important settings include:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `MODEL_ROUTER_URL` | `http://llama-router:8080/v1` | Shared OpenAI-compatible router. |
| `CONTROLLER_MODEL_NAME` | `controller` | Planner/validator/finalizer model ID. |
| `REASONING_MODEL_NAME` | `expert` | Reasoning model ID. |
| `CODER_MODEL_NAME` | `expert` | Coding model ID. |
| `VISION_MODEL_NAME` | `vision` | Vision model ID. |
| `MAX_MODEL_CONTEXT_TOKENS` | `8192` | Overall local model context target. |
| `MAX_CONTEXT_HISTORY_TOKENS` | `3072` | Bounded conversation history. |
| `PLANNER_CONTEXT_TOKENS` | `6144` | Planner input history budget. |
| `VALIDATION_CONTEXT_TOKENS` | `4096` | Validator input history budget. |
| `FINALIZER_CONTEXT_TOKENS` | `6144` | Finalizer input history budget. |
| `ADAPTIVE_FAST_PATHS` | `true` | Enables safe common-path reductions. |
| `LEGACY_EXECUTION_MODE` | `false` | Restores the full validation/rewrite path. |
| `ADAPTIVE_CONFIDENCE_THRESHOLD` | `0.75` | Minimum planner confidence for fast finalization. |
| `MODEL_QUEUE_TIMEOUT_S` | `1800` | Maximum model/image workload queue wait. |
| `CHECKPOINT_BACKEND` | `memory` | Explicitly selects `memory` or `sqlite`. |
| `CHECKPOINT_SQLITE_PATH` | `/data/checkpoints.sqlite3` | SQLite database path when selected. |
| `ENABLE_STREAMING` | `true` | Allows SSE when `stream=true`. |
| `VISION_MAX_IMAGES` | `8` | Maximum images considered per request. |
| `VISION_MAX_IMAGE_BYTES` | `8388608` | Soft per-image preprocessing limit. |
| `VISION_MAX_TOTAL_BYTES` | `33554432` | Soft total image preprocessing limit. |
| `VISION_MAX_DIMENSION` | `2048` | Maximum image dimension after preprocessing. |
| `VISION_CACHE_MAX_ITEMS` | `32` | Vision LRU cache size. |
| `VISION_CACHE_TTL_S` | `1800` | Vision cache lifetime. |

`CHECKPOINT_BACKEND=sqlite` is explicit. It raises a startup error if the
SQLite LangGraph checkpointer is not installed or configured; it does not
silently fall back to memory. The default remains memory for compatibility with
the base installation.

## Build and run

Build the repository-root Dockerfile:

```bash
docker build -t ai-orchestrator .
docker run --env-file .env -p 8001:8001 ai-orchestrator
```

Run locally from the repository root after installing the project:

```bash
python3 -m venv .venv
.venv/bin/pip install .
PYTHONPATH=src .venv/bin/uvicorn orchestrator.main:app --host 0.0.0.0 --port 8001
```

The application performs a router health check during startup and creates
shared HTTP clients for the router, Knowledge Service, SearXNG, Open WebUI,
and vision fetching. Runtime shutdown closes the lifecycle manager and all
owned transports.

## Streaming behavior

When both `stream=true` and `ENABLE_STREAMING=true`, the OpenAI endpoint first
emits an assistant role-only chunk. Internal lifecycle and specialist events
remain available through the existing event stream and are relayed as named
SSE events; they are not converted into assistant text. Finalizer output is
forwarded incrementally, followed by the existing stop chunk and `[DONE]`.

The stream replay buffer is bounded by `STREAM_REPLAY_MAX_EVENTS`, and closed
streams are retained only for a short replay window before cleanup.

## Compatibility and fallback rules

- Existing public routes, OpenAI fields, model echoing, SSE framing, and
  `[DONE]` behavior remain available.
- Request payloads retain their original multimodal data for specialist use,
  while checkpoint/debug representations keep only safe descriptors.
- Knowledge, web, vision, and image-generation failures retain usable fallback
  behavior wherever the existing route supports it.
- Image generation remains terminal and does not invoke textual validation or
  finalization after a successful image result.
- Set `LEGACY_EXECUTION_MODE=true` or disable `ADAPTIVE_FAST_PATHS` for an
  operational rollback to the full internal execution path.

## Repository layout

```text
src/orchestrator/
  api.py                 Public HTTP and SSE routes.
  http/                  HTTP request/state adapters kept separate from routes.
  settings.py            Environment-backed configuration.
  main.py                FastAPI application and lifecycle.
  graph/                 LangGraph construction and specialist nodes.
    specialists/         Capability-specific node entry points.
  controller/            Planner, validator, reasoning, and finalizer calls.
  runtime/               Lifecycle facade, router residency, GPU arbitration,
                         legacy Docker support, gateway, and metrics.
  clients/               Router, retrieval, web, and Open WebUI clients.
  context/               Conversation assembly, parsing, and evidence reuse.
  preprocessing/         Request normalization and conversation resolution.
  vision/                Image fetching, preprocessing, caching, and analysis.
  models/generation.py   Provider-neutral generation response helpers.
  streaming/             Bounded internal event streams and SSE serialization.
Dockerfile               Root container build and port-8001 launch command.
pyproject.toml            Packaging and runtime dependencies.
```
