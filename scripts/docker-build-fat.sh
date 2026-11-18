#!/usr/bin/env bash
# Build the fat image: library.json + tts-bank.tar.gz (scripts/pack.sh) baked
# in at /data at build time -- no mounted volume needed to serve the bank.
#   scripts/docker-build-fat.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f library.json ]    || { echo "error: no library.json (run scripts/pack.sh)"    >&2; exit 1; }
[ -f tts-bank.tar.gz ] || { echo "error: no tts-bank.tar.gz (run scripts/pack.sh)" >&2; exit 1; }

TAG=whizz-tts:fat
docker build --target runtime-fat -t "$TAG" .
echo "built $TAG"
