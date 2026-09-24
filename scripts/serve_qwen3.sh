#!/usr/bin/env bash
# Start the Qwen3-8B (docparse field lookup and extraction agent, tool calling) vLLM server (VLLM_IMAGE_NEW) and wait until /health is up.
# Sharing the H100 with the Paddle pipeline needs PADDLEOCR_VL_GPU_MEM=0.25 (docparse design D3).
# Usage: scripts/serve_qwen3.sh [up|down|logs]   (extra compose files via COMPOSE_EXTRA)
set -euo pipefail
cd "$(dirname "$0")/.."
files=(-f .devcontainer/compose.yaml)
[[ -n "${COMPOSE_EXTRA:-}" ]] && files+=(-f "$COMPOSE_EXTRA")
dc() { docker compose -p ocr-bench "${files[@]}" --profile docparse "$@"; }

case "${1:-up}" in
  up)   dc up -d --wait qwen3 && echo "qwen3 ready" ;;
  down) dc stop qwen3 ;;
  logs) dc logs -f qwen3 ;;
  *)    echo "usage: $0 [up|down|logs]" >&2; exit 2 ;;
esac
