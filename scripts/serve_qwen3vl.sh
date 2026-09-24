#!/usr/bin/env bash
# Start the Qwen3-VL-8B-Instruct (docparse classifier and verifier crop reads) vLLM server (VLLM_IMAGE_NEW) and wait until /health is up.
# Sharing the H100 with the Paddle pipeline needs PADDLEOCR_VL_GPU_MEM=0.25 (docparse design D3).
# Usage: scripts/serve_qwen3vl.sh [up|down|logs]   (extra compose files via COMPOSE_EXTRA)
set -euo pipefail
cd "$(dirname "$0")/.."
files=(-f .devcontainer/compose.yaml)
[[ -n "${COMPOSE_EXTRA:-}" ]] && files+=(-f "$COMPOSE_EXTRA")
dc() { docker compose -p ocr-bench "${files[@]}" --profile docparse "$@"; }

case "${1:-up}" in
  up)   dc up -d --wait qwen3vl && echo "qwen3vl ready" ;;
  down) dc stop qwen3vl ;;
  logs) dc logs -f qwen3vl ;;
  *)    echo "usage: $0 [up|down|logs]" >&2; exit 2 ;;
esac
