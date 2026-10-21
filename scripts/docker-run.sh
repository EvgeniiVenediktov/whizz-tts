#!/usr/bin/env bash
# Run an image built by docker-build-thin.sh / docker-build-fat.sh.
#   scripts/docker-run.sh thin               # whizz-tts:thin
#   scripts/docker-run.sh fat                # whizz-tts:fat (default)
#   scripts/docker-run.sh fat --with-model   # whizz-tts:fat-model
#                                            # (MODEL_ENABLED=1, PRELOAD_MODEL=1)
#
# /data is a named volume (whizz-tts-<mode>-data; override with DATA_VOLUME), so
# phrases /v1/speak generates survive restarts, as on a real persistent volume.
# A fresh volume over the fat image is seeded from the image's baked-in /data.
#
# thin: bind-mounts local library.json + tts-bank.tar.gz onto /data when both
# exist (unpacked into the volume on first boot -- local testing only).
# Without them it runs bank-less, which needs GEMINI_API_KEY or --with-model.
#
# GEMINI_API_KEY / GOOGLE_API_KEY / JWT_SECRET pass through from the
# environment (./.env is loaded first if present) by name, so values never
# appear on the docker command line. DOCKER_DNS=<ip> sets the container's
# resolver, for a local DNS that won't resolve generativelanguage.googleapis.com.
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

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

ENV_ARGS=()
[ "$WITH_MODEL" = "1" ] && ENV_ARGS+=(-e MODEL_ENABLED=1 -e PRELOAD_MODEL=1)
for var in GEMINI_API_KEY GOOGLE_API_KEY JWT_SECRET; do
    if [ -n "${!var:-}" ]; then
        export "${var?}"
        ENV_ARGS+=(-e "$var")
    fi
done

MOUNT_ARGS=(-v "${DATA_VOLUME:-whizz-tts-$MODE-data}:/data")
if [ "$MODE" = "thin" ]; then
    if [ -f library.json ] && [ -f tts-bank.tar.gz ]; then
        MOUNT_ARGS+=(-v "$PWD/library.json:/data/library.json:ro"
                     -v "$PWD/tts-bank.tar.gz:/data/tts-bank.tar.gz:ro")
    elif [ "$WITH_MODEL" = "0" ] && [ -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]; then
        echo "error: no library.json + tts-bank.tar.gz (run scripts/pack.sh), and no" >&2
        echo "       GEMINI_API_KEY or --with-model to run bank-less" >&2
        exit 1
    fi
fi

RUN_ARGS=(--rm -p 8000:8000)
if [ -t 0 ] && [ -t 1 ]; then
    RUN_ARGS+=(-it)
fi
if [ -n "${DOCKER_DNS:-}" ]; then
    RUN_ARGS+=(--dns "$DOCKER_DNS")
fi

docker run "${RUN_ARGS[@]}" "${ENV_ARGS[@]}" "${MOUNT_ARGS[@]}" "$TAG"
