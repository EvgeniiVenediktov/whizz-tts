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
#
# A container that ends up with NO usable bank AND live synthesis off can serve
# nothing, so it refuses to start (exit 1) rather than come up "healthy" and
# 404 every lookup. Exception: MODEL_ENABLED=1 is a valid bank-less deployment
# (live-synthesis-only) — /health already treats that as serving (200, not 503),
# so the entrypoint mirrors that same rule instead of hard-blocking it.
#
# Bank sync: two ways to pull the bank into the volume before/instead of the
# packaged archive.
#   - SYNC_BUCKET + SYNC_PREFIX: structured, injection-free. Runs a fixed,
#     properly-quoted `aws s3` sync — safe even if the values come from
#     untrusted input, since they're passed as argv, never interpolated into
#     a shell string.
#   - BANK_SYNC_CMD: escape hatch — runs `sh -c "$BANK_SYNC_CMD"`, i.e. an
#     ARBITRARY shell command. This is intentionally flexible (e.g. rclone,
#     gsutil, a multi-step pull) but it means whoever sets this env var
#     controls what the container executes. Only set it from trusted
#     configuration (not from anything request-derived or user-supplied) —
#     same trust boundary as the container's own command line.
# If both are set, the structured SYNC_BUCKET/SYNC_PREFIX path runs and
# BANK_SYNC_CMD is skipped (logged either way).
set -e

BANK_DB="${BANK_DB:-/data/wordbank.db}"
DATA_DIR="$(dirname "$BANK_DB")"
LIBRARY_DESC="${LIBRARY_DESC:-$DATA_DIR/library.json}"
LIBRARY_ARCHIVE="${LIBRARY_ARCHIVE:-$DATA_DIR/tts-bank.tar.gz}"

# Mirrors src/server.py's _truthy(): "" / 0 / false / no / off (case-insensitive)
# are falsy, anything else (including unset-but-defaulted "1") is truthy.
model_enabled() {
    case "$(printf '%s' "${MODEL_ENABLED:-}" | tr '[:upper:]' '[:lower:]')" in
        ''|0|false|no|off) return 1 ;;
        *) return 0 ;;
    esac
}

# Shared explanation for "nothing usable" failures.
serving_nothing_msg() {
    cat <<EOF
[entrypoint] ERROR: no usable bank and live synthesis is off (MODEL_ENABLED=${MODEL_ENABLED:-0}) — this container can serve nothing, refusing to start.
[entrypoint]   Looked for a description at: $LIBRARY_DESC
[entrypoint]   Looked for an archive at:     $LIBRARY_ARCHIVE
[entrypoint]   Fix one of:
[entrypoint]     - Put library.json + tts-bank.tar.gz on the volume (bash scripts/pack.sh
[entrypoint]       locally, then place both files at $DATA_DIR, e.g. mounted as in
[entrypoint]       docker-compose.yml) so this entrypoint can provision $DATA_DIR.
[entrypoint]     - Point LIBRARY_DESC / LIBRARY_ARCHIVE at wherever you did place them.
[entrypoint]     - Set SYNC_BUCKET/SYNC_PREFIX (or BANK_SYNC_CMD) to pull the bank from
[entrypoint]       storage before boot.
[entrypoint]     - Set MODEL_ENABLED=1 for a live-synthesis-only deployment (no bank).
EOF
}

# Bank sync runs FIRST — it is a way of *provisioning* the volume, so it has to
# happen before verification decides whether the volume is usable. (Running it
# after the check would make a sync-provisioned deployment exit 1 on an empty
# volume before the sync ever got a chance to populate it.)
if [ -n "$SYNC_BUCKET" ] && [ -n "$SYNC_PREFIX" ]; then
    if command -v aws >/dev/null 2>&1; then
        echo "[entrypoint] syncing bank via structured params: s3://$SYNC_BUCKET/$SYNC_PREFIX -> $DATA_DIR"
        mkdir -p "$DATA_DIR/words_ogg"
        aws s3 sync "s3://$SYNC_BUCKET/$SYNC_PREFIX/words_ogg" "$DATA_DIR/words_ogg"
        aws s3 cp "s3://$SYNC_BUCKET/$SYNC_PREFIX/wordbank.db" "$DATA_DIR/wordbank.db"
    else
        echo "[entrypoint] ERROR: SYNC_BUCKET/SYNC_PREFIX set but no 'aws' CLI in this image (the bank-only build ships no cloud CLI to keep it small)." >&2
        echo "[entrypoint]   Fix: use BANK_SYNC_CMD instead (it can invoke any tool you install), or build a variant image with awscli installed." >&2
        exit 1
    fi
elif [ -n "$BANK_SYNC_CMD" ]; then
    # Escape hatch: arbitrary shell command. Trusted-configuration only — see
    # header comment. Prefer SYNC_BUCKET/SYNC_PREFIX above when possible.
    echo "[entrypoint] syncing bank via BANK_SYNC_CMD (trusted-config shell command): $BANK_SYNC_CMD"
    sh -c "$BANK_SYNC_CMD"
fi

if python -m wordbank verify --desc "$LIBRARY_DESC" --root "$DATA_DIR"; then
    echo "[entrypoint] library verified — starting"
elif [ -f "$LIBRARY_ARCHIVE" ]; then
    echo "[entrypoint] volume not provisioned — unpacking $LIBRARY_ARCHIVE -> $DATA_DIR"
    mkdir -p "$DATA_DIR"
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" --strip-components=1 output/words_ogg
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" wordbank.db
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" --strip-components=1 output/voices 2>/dev/null || true
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" library.json 2>/dev/null || true
    if python -m wordbank verify --desc "$LIBRARY_DESC" --root "$DATA_DIR"; then
        echo "[entrypoint] library verified after unpack — starting"
    elif model_enabled; then
        echo "[entrypoint] WARNING: still invalid after unpacking $LIBRARY_ARCHIVE (see verification problems above) — MODEL_ENABLED is on, continuing in live-synthesis-only mode (bank unavailable)"
    else
        echo "[entrypoint] ERROR: still invalid after unpacking $LIBRARY_ARCHIVE (see verification problems above)." >&2
        echo "[entrypoint]   The archive at $LIBRARY_ARCHIVE doesn't match $LIBRARY_DESC (corrupt archive, stale description, or a failed extraction)." >&2
        echo "[entrypoint]   Fix: regenerate both with 'bash scripts/pack.sh' and re-place library.json + tts-bank.tar.gz on the volume, or set MODEL_ENABLED=1 to run live-synthesis-only." >&2
        exit 1
    fi
else
    echo "[entrypoint] volume not provisioned and no archive at $LIBRARY_ARCHIVE"
    if model_enabled; then
        echo "[entrypoint] WARNING: no bank and no archive — MODEL_ENABLED is on, continuing in live-synthesis-only mode (bank unavailable)"
    else
        serving_nothing_msg >&2
        exit 1
    fi
fi

exec "$@"
