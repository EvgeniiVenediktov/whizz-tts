#!/usr/bin/env bash
# Pack the bank server + data into ONE tarball for easy transfer & setup.
#
#   scripts/pack.sh [OUTPUT.tar.gz]        # default: tts-bank.tar.gz
#
# Bundles the library code + wordbank.db + the served OGG bank (output/words_ogg
# and output/voices). Deliberately EXCLUDES the multi-GB WAV masters under
# output/words — they're dev-only and never served. Unpack with scripts/unpack.sh.
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root
OUT="${1:-tts-bank.tar.gz}"

[ -f wordbank.db ]      || { echo "error: no wordbank.db (generate or pull the bank first)"   >&2; exit 1; }
[ -d output/words_ogg ] || { echo "error: no output/words_ogg (run scripts/encode_wavs.py)"   >&2; exit 1; }

paths=(wordbank src frontend scripts
       requirements.txt requirements-base.txt requirements-model.txt
       README.md AGENTS.md pytest.ini Dockerfile docker-compose.yml docker-entrypoint.sh
       wordbank.db output/words_ogg)
if [ -d output/voices ]; then paths+=(output/voices); fi   # clone refs (small)

n=$(find output/words_ogg -name '*.ogg' | wc -l | tr -d ' ')
echo "packing library + wordbank.db + ${n} OGG clips → ${OUT}"
COPYFILE_DISABLE=1 tar -czf "$OUT" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='.env' \
  "${paths[@]}"

echo "done: ${OUT} ($(du -h "$OUT" | cut -f1))"
echo "set up on the target:"
echo "  mkdir tts-bank && tar xzf ${OUT} -C tts-bank && cd tts-bank && bash scripts/unpack.sh"
