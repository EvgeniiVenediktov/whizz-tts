# Single Dockerfile, two flavors via --build-arg WITH_MODEL:
#   WITH_MODEL=0 (default) → lightweight bank-only server (~150 MB, no torch)
#   WITH_MODEL=1           → adds live OmniVoice synthesis (multi-GB, CPU torch)
#
# The image is CODE ONLY — it never stores the bank. The bank lives on a mounted
# volume; on start the entrypoint verifies the volume against a description file
# (library.json) and unpacks the library archive (tts-bank.tar.gz) only if the
# volume isn't already provisioned. Deployment package = image + library.json +
# tts-bank.tar.gz (the latter two placed on the volume; see docker-entrypoint.sh).
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ARG WITH_MODEL=0

# Base server deps (always).
COPY requirements-base.txt requirements-model.txt ./
RUN pip install -r requirements-base.txt

# Optional ML stack + ffmpeg, only for the model build. CPU torch from the
# dedicated wheel index keeps it as small as possible (no CUDA).
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
