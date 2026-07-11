# AI Orchestrator

Docker-native LangGraph orchestrator for the homelab AI stack.

## Included

- OpenAI-compatible chat API
- LangGraph workflow
- Knowledge Service retrieval integration
- Ollama model adapters
- MCP gateway client stub
- Open WebUI friendly response shape

## Quick start

```bash
cp .env.example .env
cd compose
docker compose up --build
```

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Notes

This repository is intentionally v1: one backend, one graph, one API.
