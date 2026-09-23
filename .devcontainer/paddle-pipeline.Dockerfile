# PaddleOCR-VL-1.6's official two-stage pipeline as an HTTP service (design D15).
#
# PaddleX's own serving app (`paddlex --serve`, POST /layout-parsing) running the pipeline config
# shipped with the installed PaddleX, `PaddleOCR-VL-1.6`, unmodified except that VL recognition
# calls our vLLM server (`genai_config.backend: vllm-server`, the documented way to use vLLM).
# Layout detection (PP-DocLayoutV3) runs where PIPELINE_DEVICE says: `cpu` with the default
# paddlepaddle CPU wheel (small image), or `gpu:0` with PADDLE_PKG=paddlepaddle-gpu from Paddle's
# CUDA 12.6 index (compose default: the CPU layout was the bottleneck on the 16-core H100 host).
# Layout time is part of the page latency the harness measures.
# Model files (PP-DocLayoutV3, ~130 MB) download on first start into /root/.paddlex (a volume).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Pins: paddleocr 3.7.0 (doc-parser extra, PaddleOCR-VL-1.6 support) and its paddlex 3.7.x.
ARG PADDLE_PKG=paddlepaddle
ARG PADDLE_INDEX=https://pypi.org/simple
RUN pip install --no-cache-dir "${PADDLE_PKG}==3.3.1" -i "${PADDLE_INDEX}" \
        --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir "paddleocr[doc-parser]==3.7.0" \
    && paddlex --install serving -y \
    && paddlex --get_pipeline_config PaddleOCR-VL-1.6 --save_path /config \
    && python -c "import yaml; yaml.safe_load(open('/config/PaddleOCR-VL-1.6.yaml'))"

COPY paddle-pipeline-entrypoint.py /usr/local/bin/paddle-pipeline-entrypoint.py

ENV VL_SERVER_URL=http://paddleocr_vl:8000/v1 PORT=8080 PIPELINE_DEVICE=cpu
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --retries=40 --start-period=60s \
    CMD curl -fs http://localhost:${PORT}/health || exit 1
ENTRYPOINT ["python", "/usr/local/bin/paddle-pipeline-entrypoint.py"]
