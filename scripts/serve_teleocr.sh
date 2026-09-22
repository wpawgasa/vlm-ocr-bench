#!/usr/bin/env bash
# Start the teleocr vLLM server and wait until /health is up.
# Usage: scripts/serve_teleocr.sh [up|down|logs]   (extra compose files via COMPOSE_EXTRA, e.g. .devcontainer/compose.v100.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
files=(-f .devcontainer/compose.yaml)
[[ -n "${COMPOSE_EXTRA:-}" ]] && files+=(-f "$COMPOSE_EXTRA")
dc() { docker compose -p ocr-bench "${files[@]}" --profile serve --profile typhoon "$@"; }

case "${1:-up}" in
  up)   dc up -d --wait teleocr && echo "teleocr ready" ;;
  down) dc stop teleocr ;;
  logs) dc logs -f teleocr ;;
  *)    echo "usage: $0 [up|down|logs]" >&2; exit 2 ;;
esac
