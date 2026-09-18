# `data/` folder

This directory is included in the Docker image (see `Dockerfile` + `.dockerignore`)
so the interactive dashboard works inside the container, not just on the host.

The 10 public sample cases live here as `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json`
and are served by `GET /api/samples`.

The original (gitignored) source copy remains at `Participant_Docs/` for local development.
