#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile -t ai-orchestrator .
docker run --rm --env-file .env -p "${APP_PORT:-8000}:8000" ai-orchestrator
