#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
docker compose -f compose/compose.yaml -f compose/compose.override.yaml up --build
