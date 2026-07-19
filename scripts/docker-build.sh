#!/usr/bin/env bash
# Build the server image via docker compose.
#   scripts/docker-build.sh          # bank-only (no torch)
#   scripts/docker-build.sh full     # with live OmniVoice synthesis
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose --profile "${1:-bank}" build
