# Zion Git Repository RAG API

Zion exposes repository search and grounded answers, with Open WebUI as its primary chat interface. Repository registration, Git synchronization, parsing, chunking, embeddings, and Qdrant writes are owned by the independent Knowledge Service. Answer generation uses the configured OpenAI-compatible llama-router endpoint.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/models` | OpenAI-compatible model discovery (`orchestrator` is the primary model ID). |
| POST | `/v1/chat/completions` | OpenAI-compatible JSON and streaming chat completions; general questions go to `expert`, repository questions use grounded retrieval. |
| GET | `/healthz`, `/readyz`, `/metrics` | Liveness, readiness, and Prometheus metrics. |

The chat endpoint accepts the previous request fields (`model`, `messages`, `stream`, `temperature`, `max_tokens`, `params`, and `metadata`), plus OpenAI `tools` and tool-call messages for Open WebUI Native Web Search. Use `X-Orchestrator-Thread-ID` or `metadata.thread_id` to preserve a conversation identity; send prior turns in `messages` for follow-up resolution. Zion classifies each turn using `expert`: general questions are answered from model knowledge without calling the Knowledge Service, while repository questions retrieve evidence and require grounding. Ambiguous questions use repository retrieval and return an insufficient-evidence answer when needed. When Open WebUI includes its `search_web` tool, Zion returns one tool call; Open WebUI executes it using its configured search provider (SearXNG) and resubmits the result for answer generation. Zion cites used web results as links and reports them separately in `metadata.web_search`; repository citations remain in `metadata.citations`. Streaming uses OpenAI SSE content/tool-call deltas and `[DONE]`, with metadata on terminal chunks.

In Open WebUI, enable the Web Search capability for the `orchestrator` model, configure SearXNG as the Web Search provider, and use Native function-calling mode. Search occurs only when the `search_web` tool is present in the request, which is how Zion detects the per-chat toggle. See [Open WebUI Agentic Web Search](https://docs.openwebui.com/features/chat-conversations/web-search/agentic-search/) for its current setup and tool flow.

Additional repository endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/repositories` | List indexed repositories (Knowledge Service v1 contract required). |
| POST | `/api/v1/repositories` | Register a repository (Knowledge Service v1 contract required). |
| GET | `/api/v1/repositories/{id}` | Repository indexing status. |
| POST | `/api/v1/repositories/{id}/sync` | Trigger incremental synchronization. |
| POST | `/api/v1/repositories/{id}/reindex` | Trigger full re-index. |
| DELETE | `/api/v1/repositories/{id}` | Remove repository and indexed data via Knowledge Service. |
| GET | `/api/v1/jobs/{id}` | Index job status. |
| POST | `/api/v1/search` | Retrieve repository context and citations. |
| POST | `/api/v1/ask` | Answer from grounded repository evidence. |

`/api/v1/ask` remains repository-grounded only. It returns `answer`, `grounded`, `confidence`, `citations`, retrieval metadata, and optionally the retrieved context. When the Knowledge Service reports no grounded evidence, Zion returns an explicit insufficient-context answer without calling the answer-generation LLM.

The repository and job operations target the Knowledge Service v1 paths documented above. They return an upstream error until that independent service implements those paths. The currently deployed `POST /retrieve` endpoint remains the compatibility path for unfiltered search and answering. Zion rejects repository or branch filters with HTTP 501 until the Knowledge Service confirms and implements filter semantics; it will not silently return cross-repository results for a scoped request.

## Configuration

Settings use the `ZION_` prefix and may be supplied through environment variables or `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ZION_API_KEY` | unset | Optional bearer credential when auth is explicitly enabled. |
| `ZION_REQUIRE_AUTH` | `false` | Optional authentication gate for Zion API routes. |
| `ZION_LLM_BASE_URL` | `http://192.168.1.61:8081/v1` | llama-router OpenAI-compatible API. |
| `ZION_LLM_MODEL` | `expert` | Answer-generation model alias. |
| `ZION_LLM_API_KEY` | unset | Optional llama-router credential. |
| `ZION_LLM_TIMEOUT_S` | `180` | LLM request timeout. |
| `ZION_KNOWLEDGE_SERVICE_URL` | `http://knowledge-service:8000` | Independent Knowledge Service. |
| `ZION_KNOWLEDGE_SERVICE_API_KEY` | unset | Optional upstream service credential. |
| `ZION_KNOWLEDGE_TIMEOUT_S` | `45` | Retrieval/management request timeout. |
| `ZION_MAX_CONTEXT_CHARS` | `24000` | Maximum retrieved context sent to the answer model. |

Embedding configuration belongs to the Knowledge Service because that service owns ingestion. Existing vector model, dimensions, collection names, and payload compatibility must be verified there before any re-indexing or migration.

## Run

```sh
python -m venv .venv
.venv/bin/pip install .
PYTHONPATH=src .venv/bin/uvicorn orchestrator.main:app --host 0.0.0.0 --port 8001
```

Build the container with `docker build -t zion-git-rag .`. The application image does not install Docker CLI or manage other containers. Zion requires network access to the Knowledge Service and llama-router. Qdrant is accessed by the Knowledge Service, not directly by Zion.

## Data and deployment boundary

This repository does not contain Knowledge Service source, Docker Compose, Qdrant configuration, Git clones, or indexed data. No Zion operation directly changes Qdrant. Before changing embedding models, collection schemas, or deleting indexed data, inventory and back up the Knowledge Service’s collections and repository state. Keep the previous deployed image/configuration available for rollback.
