#!/usr/bin/env bash
# Build the thin image: code only, no baked bank -- the bank has to come from
# a mounted /data volume at runtime (e.g. a k8s PVC), SYNC_BUCKET, or
# BANK_SYNC_CMD.
#   scripts/docker-build-thin.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=whizz-tts:thin
docker build --target runtime -t "$TAG" .
echo "built $TAG"
