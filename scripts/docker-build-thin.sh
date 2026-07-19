#!/usr/bin/env bash
# Build the thin image: code only, no baked bank -- the bank has to come from
# a mounted /data volume at runtime (e.g. a k8s PVC), SYNC_BUCKET, or
# BANK_SYNC_CMD.
#   scripts/docker-build-thin.sh              # no torch
#   scripts/docker-build-thin.sh --with-model  # + live OmniVoice synthesis (torch)
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_MODEL=0
[ "${1:-}" = "--with-model" ] && WITH_MODEL=1

TAG=whizz-tts:thin
[ "$WITH_MODEL" = "1" ] && TAG="$TAG-model"

docker build --target runtime --build-arg WITH_MODEL="$WITH_MODEL" -t "$TAG" .
echo "built $TAG"
