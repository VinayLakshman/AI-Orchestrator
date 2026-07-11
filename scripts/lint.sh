#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
docker compose -f compose/compose.yaml exec orchestrator python -m compileall app
