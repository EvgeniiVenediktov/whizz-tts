#!/bin/sh
# Optional pre-start sync: pull the OGG bank + manifest (and anything else) into
# the mounted volume before serving. Set BANK_SYNC_CMD to your storage command,
# e.g.:
#   BANK_SYNC_CMD="aws s3 sync s3://my-bucket/words_ogg /data/words_ogg \
#                  && aws s3 cp s3://my-bucket/wordbank.db /data/wordbank.db"
# (whatever CLI you use must be installed in the image). If unset, we assume the
# volume is already populated (bind mount or an external init step).
set -e

if [ -n "$BANK_SYNC_CMD" ]; then
    echo "[entrypoint] syncing bank: $BANK_SYNC_CMD"
    sh -c "$BANK_SYNC_CMD"
fi

exec "$@"
