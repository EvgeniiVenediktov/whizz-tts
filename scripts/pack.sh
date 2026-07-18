#!/usr/bin/env bash
# Build a deployment package: a library description + a library archive.
#
#   scripts/pack.sh [ARCHIVE.tar.gz]       # default: tts-bank.tar.gz
#
# Produces:
#   library.json      — description (languages, voices, word/phrase counts, folders)
#   tts-bank.tar.gz    — library code + wordbank.db + served OGG (output/words_ogg
#                       + output/voices). EXCLUDES the multi-GB WAV masters.
#
# Deployment package = the Docker image + library.json + tts-bank.tar.gz. Put the
# latter two on the (persistent) volume for the first deploy; the container
# verifies the volume against library.json and unpacks the archive if needed.
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root
OUT="${1:-tts-bank.tar.gz}"
PY="$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)"

[ -f wordbank.db ]      || { echo "error: no wordbank.db (generate or pull the bank first)"   >&2; exit 1; }
[ -d output/words_ogg ] || { echo "error: no output/words_ogg (run scripts/encode_wavs.py)"   >&2; exit 1; }

# 1. Describe the bank → library.json (also bundled into the archive).
"$PY" -m wordbank describe --output library.json

# 2. Bundle code + db + served audio + the description.
paths=(wordbank src frontend scripts
       requirements.txt requirements-base.txt requirements-model.txt
       README.md AGENTS.md pytest.ini Dockerfile docker-compose.yml docker-entrypoint.sh
       library.json wordbank.db output/words_ogg)
if [ -d output/voices ]; then paths+=(output/voices); fi

n=$(find output/words_ogg -name '*.ogg' | wc -l | tr -d ' ')
echo "packing library + wordbank.db + ${n} OGG clips → ${OUT}"
COPYFILE_DISABLE=1 tar -czf "$OUT" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='.env' \
  "${paths[@]}"

echo "done. deployment package = docker image + library.json + ${OUT} ($(du -h "$OUT" | cut -f1))"
echo "  Docker : mount library.json + ${OUT} on the volume; the container provisions on first boot"
echo "  bare   : mkdir tts-bank && tar xzf ${OUT} -C tts-bank && cd tts-bank && bash scripts/unpack.sh"
