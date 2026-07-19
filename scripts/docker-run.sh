#!/usr/bin/env bash
# Run the server via docker compose. Needs library.json + tts-bank.tar.gz in
# the repo root first (scripts/pack.sh) — mounted read-only for the entrypoint
# to provision the /data volume on first boot.
#   scripts/docker-run.sh            # bank-only, no model ever loaded
#   scripts/docker-run.sh full       # live synthesis, model preloaded at startup
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${1:-bank}"

[ -f library.json ]    || { echo "error: no library.json (run scripts/pack.sh)"    >&2; exit 1; }
[ -f tts-bank.tar.gz ] || { echo "error: no tts-bank.tar.gz (run scripts/pack.sh)" >&2; exit 1; }

docker compose --profile "$PROFILE" up "tts-$PROFILE"
