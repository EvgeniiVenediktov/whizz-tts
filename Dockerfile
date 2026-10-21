# Two flavors via --build-arg WITH_MODEL, and two --target stages for how the
# bank gets onto the container:
#   WITH_MODEL=0 (default) → lightweight bank server, no torch: serves the bank
#                            and fills misses via Gemini (/v1/speak)
#   WITH_MODEL=1           → adds live OmniVoice synthesis (multi-GB, CPU torch)
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
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ARG WITH_MODEL=0

# Base server deps (always). src/audio.py calls `ffmpeg` from PATH, so expose
# imageio-ffmpeg's static binary there (/v1/speak encodes Gemini output to OGG).
COPY requirements-base.txt requirements-model.txt ./
RUN pip install -r requirements-base.txt && \
    ln -s "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
        /usr/local/bin/ffmpeg && \
    ffmpeg -hide_banner -encoders | grep -q libopus

# Optional ML stack, only for the model build. (Its apt ffmpeg predates the
# base layer's static one; kept so the model build's audio decoding is
# unchanged.) CPU torch from the dedicated wheel index keeps it as small as
# possible (no CUDA).
RUN if [ "$WITH_MODEL" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
        rm -rf /var/lib/apt/lists/* && \
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
        pip install -r requirements-model.txt ; \
    fi

COPY src/ ./src/
COPY wordbank/ ./wordbank/
COPY frontend/ ./frontend/
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

ENV MODEL_ENABLED=0 \
    SERVE_FRONTEND=1 \
    BANK_DB=/data/wordbank.db \
    OGG_ROOT=/data/words_ogg \
    VOICES_ROOT=/data/voices

EXPOSE 8000

# Wired to GET /health, which returns 503 when the service can serve nothing
# (bank unavailable AND live synthesis disabled) and 200 otherwise. No curl/wget
# in this base image (python:3.12-slim) and the bank-only build has no ML stack,
# so use a stdlib-only check: urllib raises on any non-2xx status (including the
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
COPY library.json tts-bank.tar.gz /data/

# `runtime` is a plain alias for `base`, kept as the LAST stage in this file
# so it stays the implicit default target (`docker build .` with no --target)
# — i.e. the default build stays code-only, matching prior behavior.
FROM base AS runtime
