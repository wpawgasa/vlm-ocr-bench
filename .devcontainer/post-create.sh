#!/usr/bin/env bash
set -euo pipefail

# Named volumes are created root-owned on first use.
sudo chown -R vscode:vscode /home/vscode/.venv /home/vscode/.cache/uv

if [[ -f pyproject.toml ]]; then
  uv sync --all-extras
else
  echo "post-create: no pyproject.toml yet (milestone M0) — run 'uv sync --all-extras' once it exists."
fi

# Spec-driven workflow CLI; `openspec init` output lives in the repo.
npm install -g @fission-ai/openspec@latest

echo "--- toolchain ---"
python --version
uv --version
node --version
openspec --version
pdftoppm -v 2>&1 | head -1
fc-list :lang=th family | head -3
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable (latency sampler needs it on the GPU host)"
