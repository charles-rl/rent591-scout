#!/usr/bin/env bash
# Periodic local ingest: sync relay payloads from GitHub, then run the hybrid
# pipeline. Pull failure (offline / diverged) is logged but never blocks the
# run — --incoming is idempotent via relay_state, so processing the payloads
# already on disk is always safe.
set -u
cd "$(dirname "$0")/.."

if ! git pull --ff-only origin "$(git rev-parse --abbrev-ref HEAD)"; then
  echo "WARN: git pull failed (offline or diverged) - ingesting existing payloads" >&2
fi

exec .venv/bin/python main.py --incoming
