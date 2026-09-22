# Harness container for ocr-bench.
# The harness never loads model weights (spec §4): models are served by the
# vLLM services in compose.yaml, so this image needs no CUDA toolkit.
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# poppler-utils: pdffonts / pdftoppm / pdftotext for bank-statement ingestion (spec §3.2)
# fonts-thai-tlwg: Thai glyphs for plots and rendered samples in the report
# libgl1 / libglib2.0-0: runtime deps for opencv (degradation transforms, §3.3)
# jq: inspecting runs/<run_id>/*.jsonl
# The base image ships a Yarn apt source with an expired key; we don't need it.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        fonts-thai-tlwg \
        fontconfig \
        libgl1 \
        libglib2.0-0 \
        build-essential \
        jq \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /usr/local/bin/

# Project venv lives outside the bind-mounted workspace so it survives
# rebuilds of the workspace and never collides with a host-side .venv.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PROJECT_ENVIRONMENT=/home/vscode/.venv \
    VIRTUAL_ENV=/home/vscode/.venv \
    PATH=/home/vscode/.venv/bin:${PATH} \
    PYTHONUNBUFFERED=1

RUN mkdir -p /home/vscode/.venv /home/vscode/.cache/uv /home/vscode/.cache/huggingface \
    && chown -R vscode:vscode /home/vscode/.venv /home/vscode/.cache
