#!/bin/sh
# Provision the bank volume, then exec the server. In order:
#   1. pull the bank (SYNC_BUCKET/SYNC_PREFIX, or BANK_SYNC_CMD) — first, since
#      it provisions the volume the next step checks;
#   2. verify the volume against LIBRARY_DESC (library.json);
#   3. if invalid, unpack LIBRARY_ARCHIVE (tts-bank.tar.gz) once and re-verify;
#   4. with no usable bank and no synthesis (GEMINI_API_KEY/GOOGLE_API_KEY, or
#      SPEAK_UPSTREAM_URL), refuse to start — /health applies the same rule.
# A replaced wordbank.db is moved aside with its -wal/-shm, never overwritten.
# BANK_SYNC_CMD runs via `sh -c`: set it from trusted configuration only (the
# same trust as the container's command line).
set -e

BANK_DB="${BANK_DB:-/data/wordbank.db}"
DATA_DIR="$(dirname "$BANK_DB")"
LIBRARY_DESC="${LIBRARY_DESC:-$DATA_DIR/library.json}"
LIBRARY_ARCHIVE="${LIBRARY_ARCHIVE:-$DATA_DIR/tts-bank.tar.gz}"
UNPACK_FAILED="$DATA_DIR/.unpack-failed"   # cksum of an archive that failed verification once unpacked

can_synthesize() {
    [ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}${SPEAK_UPSTREAM_URL:-}" ]
}

move_db_aside() {      # $1: why (unpack | sync)
    [ -f "$BANK_DB" ] || return 0
    backup="$(mktemp "$BANK_DB.pre-$1-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
    for ext in "" -wal -shm; do
        if [ -f "$BANK_DB$ext" ]; then mv "$BANK_DB$ext" "$backup$ext"; fi
    done
    echo "[entrypoint] existing $BANK_DB moved to $backup before the $1"
}

# Shared explanation for "nothing usable" failures.
serving_nothing_msg() {
    cat <<EOF
[entrypoint] ERROR: no usable bank and no synthesis (no GEMINI_API_KEY, no SPEAK_UPSTREAM_URL) — this container can serve nothing, refusing to start.
[entrypoint]   Looked for a description at: $LIBRARY_DESC
[entrypoint]   Looked for an archive at:     $LIBRARY_ARCHIVE
[entrypoint]   Fix one of:
[entrypoint]     - Put library.json + tts-bank.tar.gz on the volume (bash scripts/pack.sh
[entrypoint]       locally, then place both files at $DATA_DIR, e.g. mounted as in
[entrypoint]       docker-compose.yml) so this entrypoint can provision $DATA_DIR.
[entrypoint]     - Point LIBRARY_DESC / LIBRARY_ARCHIVE at wherever you did place them.
[entrypoint]     - Set SYNC_BUCKET/SYNC_PREFIX (or BANK_SYNC_CMD) to pull the bank from
[entrypoint]       storage before boot.
[entrypoint]     - Set GEMINI_API_KEY or SPEAK_UPSTREAM_URL for a synthesis-only deployment (no bank).
EOF
}

if [ -n "$SYNC_BUCKET" ] && [ -n "$SYNC_PREFIX" ]; then
    if command -v aws >/dev/null 2>&1; then
        echo "[entrypoint] syncing bank via structured params: s3://$SYNC_BUCKET/$SYNC_PREFIX -> $DATA_DIR"
        mkdir -p "$DATA_DIR/words_ogg"
        aws s3 sync "s3://$SYNC_BUCKET/$SYNC_PREFIX/words_ogg" "$DATA_DIR/words_ogg"
        aws s3 cp "s3://$SYNC_BUCKET/$SYNC_PREFIX/wordbank.db" "$BANK_DB.sync"
        move_db_aside sync
        mv "$BANK_DB.sync" "$BANK_DB"
    else
        echo "[entrypoint] ERROR: SYNC_BUCKET/SYNC_PREFIX set but no 'aws' CLI in this image (the bank-only build ships no cloud CLI to keep it small)." >&2
        echo "[entrypoint]   Fix: use BANK_SYNC_CMD instead (it can invoke any tool you install), or build a variant image with awscli installed." >&2
        exit 1
    fi
elif [ -n "$BANK_SYNC_CMD" ]; then
    echo "[entrypoint] syncing bank via BANK_SYNC_CMD (trusted-config shell command)"
    sh -c "$BANK_SYNC_CMD"
fi

if python -m wordbank verify --desc "$LIBRARY_DESC" --root "$DATA_DIR"; then
    echo "[entrypoint] library verified — starting"
elif [ -f "$LIBRARY_ARCHIVE" ] && [ "$(cksum < "$LIBRARY_ARCHIVE")" != "$(cat "$UNPACK_FAILED" 2>/dev/null)" ]; then
    echo "[entrypoint] volume not provisioned — unpacking $LIBRARY_ARCHIVE -> $DATA_DIR"
    mkdir -p "$DATA_DIR"
    # e.g. phrases /v1/speak saved before the archive was added: kept in the backup
    move_db_aside unpack
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" --strip-components=1 output/words_ogg
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" wordbank.db
    tar xzf "$LIBRARY_ARCHIVE" -C "$DATA_DIR" library.json 2>/dev/null || true
    if python -m wordbank verify --desc "$LIBRARY_DESC" --root "$DATA_DIR"; then
        echo "[entrypoint] library verified after unpack — starting"
    # failed: recorded, so this archive isn't unpacked again on every boot
    elif cksum < "$LIBRARY_ARCHIVE" > "$UNPACK_FAILED" && can_synthesize; then
        echo "[entrypoint] WARNING: still invalid after unpacking $LIBRARY_ARCHIVE (see verification problems above) — synthesis is available (GEMINI_API_KEY / SPEAK_UPSTREAM_URL), continuing in synthesis-only mode (bank unavailable)"
    else
        echo "[entrypoint] ERROR: still invalid after unpacking $LIBRARY_ARCHIVE (see verification problems above)." >&2
        echo "[entrypoint]   The archive at $LIBRARY_ARCHIVE doesn't match $LIBRARY_DESC (corrupt archive, stale description, or a failed extraction)." >&2
        echo "[entrypoint]   Fix: regenerate both with 'bash scripts/pack.sh' and re-place library.json + tts-bank.tar.gz on the volume, or set GEMINI_API_KEY to run synthesis-only." >&2
        exit 1
    fi
elif [ ! -f "$LIBRARY_DESC" ] && [ -f "$BANK_DB" ] && can_synthesize; then
    echo "[entrypoint] no library description at $LIBRARY_DESC — serving the bank built at runtime ($BANK_DB) + synthesis for misses"
else
    if [ -f "$LIBRARY_ARCHIVE" ]; then
        echo "[entrypoint] $LIBRARY_ARCHIVE failed verification when unpacked before; not unpacking it again (delete $UNPACK_FAILED to retry)"
    else
        echo "[entrypoint] volume not provisioned and no archive at $LIBRARY_ARCHIVE"
    fi
    if can_synthesize; then
        echo "[entrypoint] WARNING: no bank and no archive — synthesis is available (GEMINI_API_KEY / SPEAK_UPSTREAM_URL), continuing in synthesis-only mode (bank unavailable)"
    else
        serving_nothing_msg >&2
        exit 1
    fi
fi

exec "$@"
