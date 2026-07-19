#!/usr/bin/env bash
# Run a server image built by scripts/docker-build.sh — plain `docker run`,
# no compose, no persistent volume (that's wired up separately once real
# storage exists, e.g. a k8s PVC).
#
#   scripts/docker-run.sh                # run whizz-tts:runtime-fat
#   scripts/docker-run.sh --with-model    # run whizz-tts:runtime-fat-model
#                                         # (MODEL_ENABLED=1, PRELOAD_MODEL=1)
#   scripts/docker-run.sh --thin          # run whizz-tts:runtime, bind-mounting
#                                         # local library.json + tts-bank.tar.gz
#                                         # at /data -- a stand-in for a real
#                                         # volume mount, local testing only
#   scripts/docker-run.sh --image NAME    # override the image tag
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=runtime-fat
WITH_MODEL=0
IMAGE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --thin) TARGET=runtime ;;
        --with-model) WITH_MODEL=1 ;;
        --image) IMAGE="$2"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

if [ -z "$IMAGE" ]; then
    IMAGE="whizz-tts:$TARGET"
    [ "$WITH_MODEL" = "1" ] && IMAGE="$IMAGE-model"
fi

ENV_ARGS=()
[ "$WITH_MODEL" = "1" ] && ENV_ARGS+=(-e MODEL_ENABLED=1 -e PRELOAD_MODEL=1)

MOUNT_ARGS=()
if [ "$TARGET" = "runtime" ]; then
    [ -f library.json ]    || { echo "error: no library.json (run scripts/pack.sh)"    >&2; exit 1; }
    [ -f tts-bank.tar.gz ] || { echo "error: no tts-bank.tar.gz (run scripts/pack.sh)" >&2; exit 1; }
    MOUNT_ARGS+=(-v "$PWD/library.json:/data/library.json:ro" -v "$PWD/tts-bank.tar.gz:/data/tts-bank.tar.gz:ro")
fi

docker run --rm -it -p 8000:8000 "${ENV_ARGS[@]}" "${MOUNT_ARGS[@]}" "$IMAGE"
