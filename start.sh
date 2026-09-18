#!/usr/bin/env bash
# Render start script.
#
# Why bash wrapper instead of a direct uvicorn command?
#   - Render injects $PORT at runtime; we honor it (uvicorn reads --port).
#   - We bind 0.0.0.0 so Render's reverse proxy can reach us.
#   - Single worker is optimal on free plan (512 MB RAM); multi-worker would
#     OOM quickly. The async event loop + asyncio.to_thread already handles
#     concurrent optimization requests within a single worker.
#   - Proxy headers are honored via --proxy-headers so client IPs are correct
#     in the access logs.
set -e

exec uvicorn main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips="*" \
  --log-level info
