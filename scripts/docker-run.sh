#!/usr/bin/env bash
# Run an image built by docker-build-thin.sh / docker-build-fat.sh.
#   scripts/docker-run.sh thin              # whizz-tts:thin, bind-mounting local
#                                            # library.json + tts-bank.tar.gz at
#                                            # /data (stand-in for a real volume
#                                            # mount -- local testing only)
#   scripts/docker-run.sh fat               # whizz-tts:fat (default)
#   scripts/docker-run.sh fat --with-model   # whizz-tts:fat-model
#                                            # (MODEL_ENABLED=1, PRELOAD_MODEL=1)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-fat}"
case "$MODE" in
    thin|fat) ;;
    *) echo "usage: $0 [thin|fat] [--with-model]" >&2; exit 1 ;;
esac

WITH_MODEL=0
[ "${2:-}" = "--with-model" ] && WITH_MODEL=1

TAG="whizz-tts:$MODE"
[ "$WITH_MODEL" = "1" ] && TAG="$TAG-model"

ENV_ARGS=()
[ "$WITH_MODEL" = "1" ] && ENV_ARGS+=(-e MODEL_ENABLED=1 -e PRELOAD_MODEL=1)

MOUNT_ARGS=()
if [ "$MODE" = "thin" ]; then
    [ -f library.json ]    || { echo "error: no library.json (run scripts/pack.sh)"    >&2; exit 1; }
    [ -f tts-bank.tar.gz ] || { echo "error: no tts-bank.tar.gz (run scripts/pack.sh)" >&2; exit 1; }
    MOUNT_ARGS+=(-v "$PWD/library.json:/data/library.json:ro" -v "$PWD/tts-bank.tar.gz:/data/tts-bank.tar.gz:ro")
fi

docker run --rm -it -p 8000:8000 "${ENV_ARGS[@]}" "${MOUNT_ARGS[@]}" "$TAG"
