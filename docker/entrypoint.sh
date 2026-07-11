#!/usr/bin/env sh
set -eu
exec uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT:-8000}"
