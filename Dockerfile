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
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
