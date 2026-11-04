#!/usr/bin/env bash
# Run an image built by docker-build-thin.sh / docker-build-fat.sh.
#   scripts/docker-run.sh thin               # whizz-tts:thin
#   scripts/docker-run.sh fat                # whizz-tts:fat (default)
#   scripts/docker-run.sh fat --with-model   # whizz-tts:fat-model
#                                            # (MODEL_ENABLED=1, PRELOAD_MODEL=1)
#   ENV_FILE=.env.edge scripts/docker-run.sh edge
#                                            # whizz-tts:thin as an edge cache: its
#                                            # own bank, misses fetched from
#                                            # SPEAK_UPSTREAM_URL (no Gemini key)
#
# /data is a named volume (whizz-tts-<mode>-data; override with DATA_VOLUME), so
# phrases /v1/speak saves survive restarts, as on a real persistent volume.
# A fresh volume over the fat image is seeded from the image's baked-in /data.
#
# thin / edge: bind-mount local library.json + tts-bank.tar.gz onto /data when
# both exist (unpacked into the volume on first boot). Without them thin runs
# bank-less (needs GEMINI_API_KEY or --with-model) and edge starts empty.
#
# Settings come from ENV_FILE (default ./.env, loaded if present) and the
# environment, and pass into the container by name, so values never appear on
# the docker command line: GEMINI_API_KEY / GOOGLE_API_KEY / JWT_SECRET, and
# for edge JWT_SECRET / SPEAK_UPSTREAM_URL / SPEAK_UPSTREAM_TOKEN /
# SPEAK_UPSTREAM_TIMEOUT instead (an edge never holds the Gemini key).
# HOST_PORT (default 8000) is the published port; DOCKER_DNS=<ip> sets the
# container's resolver, for a local DNS that won't resolve the Gemini host.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-fat}"
case "$MODE" in
    thin|fat|edge) ;;
    *) echo "usage: $0 [thin|fat|edge] [--with-model]" >&2; exit 1 ;;
esac

WITH_MODEL=0
[ "${2:-}" = "--with-model" ] && WITH_MODEL=1
if [ "$MODE" = "edge" ] && [ "$WITH_MODEL" = "1" ]; then
    echo "error: edge fetches misses from its upstream; --with-model doesn't apply" >&2
    exit 1
fi

IMAGE_MODE="$MODE"
[ "$MODE" = "edge" ] && IMAGE_MODE=thin
TAG="whizz-tts:$IMAGE_MODE"
[ "$WITH_MODEL" = "1" ] && TAG="$TAG-model"

ENV_FILE="${ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then
    case "$ENV_FILE" in
        */*) env_src="$ENV_FILE" ;;
        *) env_src="./$ENV_FILE" ;;    # bare name: `.` would search PATH
    esac
    set -a
    . "$env_src"
    set +a
fi

PASS_VARS=(GEMINI_API_KEY GOOGLE_API_KEY JWT_SECRET SPEAK_STORE)
if [ "$MODE" = "edge" ]; then
    PASS_VARS=(JWT_SECRET SPEAK_UPSTREAM_URL SPEAK_UPSTREAM_TOKEN SPEAK_UPSTREAM_TIMEOUT)
    if [ -z "${SPEAK_UPSTREAM_URL:-}" ]; then
        echo "error: edge needs SPEAK_UPSTREAM_URL (and SPEAK_UPSTREAM_TOKEN) in $ENV_FILE or the environment" >&2
        exit 1
    fi
fi

ENV_ARGS=()
[ "$WITH_MODEL" = "1" ] && ENV_ARGS+=(-e MODEL_ENABLED=1 -e PRELOAD_MODEL=1)
for var in "${PASS_VARS[@]}"; do
    if [ -n "${!var:-}" ]; then
        export "${var?}"
        ENV_ARGS+=(-e "$var")
    fi
done

MOUNT_ARGS=(-v "${DATA_VOLUME:-whizz-tts-$MODE-data}:/data")
if [ "$MODE" != "fat" ]; then
    if [ -f library.json ] && [ -f tts-bank.tar.gz ]; then
        MOUNT_ARGS+=(-v "$PWD/library.json:/data/library.json:ro"
                     -v "$PWD/tts-bank.tar.gz:/data/tts-bank.tar.gz:ro")
    elif [ "$MODE" = "thin" ] && [ "$WITH_MODEL" = "0" ] \
            && [ -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]; then
        echo "error: no library.json + tts-bank.tar.gz (run scripts/pack.sh), and no" >&2
        echo "       GEMINI_API_KEY or --with-model to run bank-less" >&2
        exit 1
    fi
fi

RUN_ARGS=(--rm -p "${HOST_PORT:-8000}:8000")
if [ -t 0 ] && [ -t 1 ]; then
    RUN_ARGS+=(-it)
fi
if [ -n "${DOCKER_DNS:-}" ]; then
    RUN_ARGS+=(--dns "$DOCKER_DNS")
fi

docker run "${RUN_ARGS[@]}" "${ENV_ARGS[@]}" "${MOUNT_ARGS[@]}" "$TAG"
