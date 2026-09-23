#!/usr/bin/env bash
# Start PaddleOCR-VL-1.6: its vLLM server (VLLM_IMAGE_NEW) plus the PaddleX pipeline server that
# calls it (builds ocr-bench/paddle-pipeline on first use), and wait until both are healthy.
# Usage: scripts/serve_paddleocr_vl.sh [up|down|logs]   (extra compose files via COMPOSE_EXTRA)
set -euo pipefail
cd "$(dirname "$0")/.."
files=(-f .devcontainer/compose.yaml)
[[ -n "${COMPOSE_EXTRA:-}" ]] && files+=(-f "$COMPOSE_EXTRA")
dc() { docker compose -p ocr-bench "${files[@]}" --profile paddle "$@"; }

case "${1:-up}" in
  up)   dc up -d --wait paddleocr_vl paddle_pipeline && echo "paddleocr_vl + paddle_pipeline ready" ;;
  down) dc stop paddle_pipeline paddleocr_vl ;;
  logs) dc logs -f paddleocr_vl paddle_pipeline ;;
  *)    echo "usage: $0 [up|down|logs]" >&2; exit 2 ;;
esac
