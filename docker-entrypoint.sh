#!/bin/sh
# Deploy-time bank provisioning, then start the server.
#
#   1. Read the library description on the volume (LIBRARY_DESC, default
#      /data/library.json) — languages, voices, word/phrase counts, folders.
#   2. Verify the volume against it (described folders exist + counts match).
#   3. If the volume isn't provisioned, unpack the library archive if present
#      (LIBRARY_ARCHIVE, default /data/tts-bank.tar.gz), then re-verify.
#   4. exec the server.
#
# So the deployment package is: this image + library.json + tts-bank.tar.gz, with
# the latter two placed on the (persistent) volume for the first deploy. On later
# restarts the volume is already provisioned, so nothing is unpacked.
# BANK_SYNC_CMD is still honored as an alternative storage-pull step.
set -e

BANK_DB="${BANK_DB:-/data/wordbank.db}"
DATA_DIR="$(dirname "$BANK_DB")"
LIBRARY_DESC="${LIBRARY_DESC:-$DATA_DIR/library.json}"
LIBRARY_ARCHIVE="${LIBRARY_ARCHIVE:-$DATA_DIR/tts-bank.tar.gz}"

if python -m wordbank verify --desc "$LIBRARY_DESC" --root "$DATA_DIR"; then
    echo "[entrypoint] library verified — starting"
elif [ -f "$LIBRARY_ARCHIVE" ]; then
    echo "[entrypoint] volume not provisioned — unpacking $LIBRARY_ARCHIVE → $DATA_DIR"
    mkdir -p "$DATA_DIR"
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" --strip-components=1 output/words_ogg
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" wordbank.db
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" --strip-components=1 output/voices 2>/dev/null || true
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" library.json 2>/dev/null || true
    python -m wordbank verify --desc "$LIBRARY_DESC" --root "$DATA_DIR" \
        || echo "[entrypoint] WARNING: still invalid after unpack — serving anyway"
else
    echo "[entrypoint] volume not provisioned and no archive at $LIBRARY_ARCHIVE"
fi

if [ -n "$BANK_SYNC_CMD" ]; then
    echo "[entrypoint] syncing bank: $BANK_SYNC_CMD"
    sh -c "$BANK_SYNC_CMD"
fi

exec "$@"
