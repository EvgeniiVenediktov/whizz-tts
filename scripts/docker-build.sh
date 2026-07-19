#!/usr/bin/env bash
# Build the server image directly with `docker build` — no docker-compose;
# that's dev-only convenience, not what ships to production.
#
#   scripts/docker-build.sh              # fat: bank baked in, no torch
#   scripts/docker-build.sh --with-model  # fat + live OmniVoice synthesis (torch)
#   scripts/docker-build.sh --thin        # code only, no baked bank (needs a
#                                          # volume mounted at /data at runtime
#                                          # -- e.g. a k8s PVC)
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=runtime-fat
WITH_MODEL=0
for arg in "$@"; do
    case "$arg" in
        --thin) TARGET=runtime ;;
        --with-model) WITH_MODEL=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

if [ "$TARGET" = "runtime-fat" ]; then
    [ -f library.json ]    || { echo "error: no library.json (run scripts/pack.sh)"    >&2; exit 1; }
    [ -f tts-bank.tar.gz ] || { echo "error: no tts-bank.tar.gz (run scripts/pack.sh)" >&2; exit 1; }
fi

TAG="whizz-tts:$TARGET"
[ "$WITH_MODEL" = "1" ] && TAG="$TAG-model"

docker build --target "$TARGET" --build-arg WITH_MODEL="$WITH_MODEL" -t "$TAG" .
echo "built $TAG"
