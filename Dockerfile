# Lightweight bank server: serves the bank and fills misses via Gemini
# (/v1/speak). Two --target stages for how the bank gets onto the container:
#
#   --target runtime (default) → CODE ONLY, never stores the bank. The bank
#     lives on a mounted volume; on start the entrypoint verifies the volume
#     against a description file (library.json) and unpacks the library
#     archive (tts-bank.tar.gz) only if the volume isn't already provisioned.
#     Deployment package = image + library.json + tts-bank.tar.gz (the latter
#     two placed on the volume; see docker-entrypoint.sh). Use once persistent
#     storage is wired up (e.g. a k8s PVC).
#   --target runtime-fat → same image, but library.json + tts-bank.tar.gz are
#     baked in at /data at build time. docker-entrypoint.sh is unchanged — it
#     still runs its verify/unpack, it just finds the archive already sitting
#     at /data instead of on a mounted volume. For deploying before
#     persistent storage exists; every rollout re-bakes the bank in.

# ---- ffmpeg: a static build with just what src/audio.py uses (s16le in,
# libopus, Ogg out): ~6 MB instead of imageio-ffmpeg's 165 MB. ~45 s cold,
# cached after.
FROM debian:trixie-slim AS ffmpeg
ARG FFMPEG_VERSION=7.1.1
ARG FFMPEG_SHA256=733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libopus-dev curl ca-certificates xz-utils && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSLo ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" && \
    echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c - && \
    tar xJf ffmpeg.tar.xz --strip-components=1 && \
    ./configure --disable-everything --disable-autodetect --disable-doc --disable-debug \
        --disable-network --disable-ffplay --disable-ffprobe --disable-avdevice --disable-swscale \
        --disable-x86asm --enable-small --enable-libopus --enable-encoder=libopus \
        --enable-demuxer=pcm_s16le --enable-decoder=pcm_s16le --enable-muxer=ogg \
        --enable-protocol=pipe,file --enable-filter=aresample,aformat,anull \
        --pkg-config-flags=--static --extra-ldflags=-static && \
    make -j"$(nproc)" ffmpeg && strip ffmpeg

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Base server deps (always); pip itself isn't needed at runtime. src/audio.py
# calls `ffmpeg` from PATH (/v1/speak encodes Gemini output to OGG).
COPY --from=ffmpeg /src/ffmpeg /usr/local/bin/ffmpeg
COPY requirements-base.txt ./
RUN pip install -r requirements-base.txt && \
    pip uninstall -y -q pip && \
    ffmpeg -hide_banner -encoders | grep -q libopus

COPY src/ ./src/
COPY wordbank/ ./wordbank/
COPY frontend/ ./frontend/
COPY docker-entrypoint.sh ./
# Code stays root-owned (read-only to the server); only /data belongs to the
# unprivileged `app` user (uid 1000) the server runs as. Bytecode is compiled
# here because `app` can't write __pycache__ into /app.
RUN chmod +x docker-entrypoint.sh && \
    python -m compileall -q src wordbank && \
    useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin app && \
    mkdir -p /data && chown app:app /data
USER app

ENV SERVE_FRONTEND=1 \
    BANK_DB=/data/wordbank.db \
    OGG_ROOT=/data/words_ogg

EXPOSE 8000

# Wired to GET /health, which returns 503 when the service can serve nothing
# (no bank, no Gemini key, no upstream) and 200 otherwise. No curl/wget in this
# base image (python:3.12-slim), so use a stdlib-only check: urllib raises on any non-2xx status (including the
# 503 above), which makes `python -c` exit non-zero — no explicit status parsing
# needed.
#   --start-period=300s: first boot may need to unpack a ~129 MB archive of
#     ~48,000 small OGG files (docker-entrypoint.sh) before the server even
#     starts listening. Locally that extraction is ~1s on fast disk, but many
#     tiny-file creates can take minutes on network-backed volumes (EBS/EFS/NFS)
#     — 5 minutes gives real headroom without permanently marking a legitimately
#     provisioning container unhealthy (failures during start-period don't count
#     toward --retries or flip the container unhealthy).
#   --interval=30s / --timeout=5s: /health does a lightweight SQLite count, not
#     I/O-heavy — 5s is generous slack, 30s keeps check overhead low.
#   --retries=3: requires 3 consecutive failures (~90s) after start-period before
#     the container is marked unhealthy, absorbing transient blips without
#     masking a real outage for long.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- runtime-fat: bank baked into the image, no external volume needed ----
# `library.json` + `tts-bank.tar.gz` (scripts/pack.sh) must exist in the build
# context. Everything else — verify, unpack, HEALTHCHECK, entrypoint — is
# identical to `base`.
FROM base AS runtime-fat
COPY --chown=app:app library.json tts-bank.tar.gz /data/

# `runtime` is a plain alias for `base`, kept as the LAST stage in this file
# so it stays the implicit default target (`docker build .` with no --target)
# — i.e. the default build stays code-only, matching prior behavior.
FROM base AS runtime
