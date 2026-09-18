"""
Vercel serverless entry point for GridWise LLM.

Why this file exists:
  - Vercel's @vercel/python runtime expects a Python module at api/index.py
    that exports a WSGI/ASGI `app` variable.
  - Our core app lives in main.py (FastAPI instance). We re-export it here
    so Vercel picks it up while the source of truth stays in main.py.

Cold-start note:
  - Vercel runs each request in a fresh serverless container; the singleton
    httpx client and LRU cache reset on cold start. First request after idle
    takes ~3-6 s (numpy/scipy import + LP solve). Subsequent warm requests
    run in <500 ms.
  - This is fine for hackathon judging but means you may want an uptime
    monitor (UptimeRobot) hitting /health every 5 minutes to keep it warm.
"""
from main import app  # noqa: F401  (Vercel expects `app` to be importable)
