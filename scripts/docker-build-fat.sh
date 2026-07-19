#!/usr/bin/env bash
# Build the fat image: library.json + tts-bank.tar.gz (scripts/pack.sh) baked
# in at /data at build time -- no mounted volume needed to serve the bank.
#   scripts/docker-build-fat.sh              # no torch
#   scripts/docker-build-fat.sh --with-model  # + live OmniVoice synthesis (torch)
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f library.json ]    || { echo "error: no library.json (run scripts/pack.sh)"    >&2; exit 1; }
[ -f tts-bank.tar.gz ] || { echo "error: no tts-bank.tar.gz (run scripts/pack.sh)" >&2; exit 1; }

WITH_MODEL=0
[ "${1:-}" = "--with-model" ] && WITH_MODEL=1

TAG=whizz-tts:fat
[ "$WITH_MODEL" = "1" ] && TAG="$TAG-model"

docker build --target runtime-fat --build-arg WITH_MODEL="$WITH_MODEL" -t "$TAG" .
echo "built $TAG"
