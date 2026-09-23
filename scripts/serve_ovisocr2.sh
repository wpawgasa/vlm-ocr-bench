#!/usr/bin/env bash
# Start the OvisOCR2 vLLM server (VLLM_IMAGE_NEW, default vllm/vllm-openai:v0.22.1) and wait until /health is up.
# Usage: scripts/serve_ovisocr2.sh [up|down|logs]   (extra compose files via COMPOSE_EXTRA)
set -euo pipefail
cd "$(dirname "$0")/.."
files=(-f .devcontainer/compose.yaml)
[[ -n "${COMPOSE_EXTRA:-}" ]] && files+=(-f "$COMPOSE_EXTRA")
dc() { docker compose -p ocr-bench "${files[@]}" --profile ovis "$@"; }

case "${1:-up}" in
  up)   dc up -d --wait ovisocr2 && echo "ovisocr2 ready" ;;
  down) dc stop ovisocr2 ;;
  logs) dc logs -f ovisocr2 ;;
  *)    echo "usage: $0 [up|down|logs]" >&2; exit 2 ;;
esac
