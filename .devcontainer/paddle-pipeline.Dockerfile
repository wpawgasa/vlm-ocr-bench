# PaddleOCR-VL-1.6's official two-stage pipeline as an HTTP service (design D15).
#
# PaddleX's own serving app (`paddlex --serve`, POST /layout-parsing) running the pipeline config
# shipped with the installed PaddleX, `PaddleOCR-VL-1.6`, unmodified except that VL recognition
# calls our vLLM server (`genai_config.backend: vllm-server`, the documented way to use vLLM).
# Layout detection (PP-DocLayoutV3) runs on CPU with paddlepaddle's CPU wheel, which keeps the
# GPU for vLLM and this image small; that time is part of the page latency the harness measures.
# Model files (PP-DocLayoutV3, ~130 MB) download on first start into /root/.paddlex (a volume).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Pins: paddleocr 3.7.0 (doc-parser extra, PaddleOCR-VL-1.6 support) and its paddlex 3.7.x.
RUN pip install --no-cache-dir paddlepaddle==3.3.1 "paddleocr[doc-parser]==3.7.0" \
    && paddlex --install serving -y \
    && paddlex --get_pipeline_config PaddleOCR-VL-1.6 --save_path /config \
    && python -c "import yaml; yaml.safe_load(open('/config/PaddleOCR-VL-1.6.yaml'))"

COPY paddle-pipeline-entrypoint.py /usr/local/bin/paddle-pipeline-entrypoint.py

ENV VL_SERVER_URL=http://paddleocr_vl:8000/v1 PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --retries=40 --start-period=60s \
    CMD curl -fs http://localhost:${PORT}/health || exit 1
ENTRYPOINT ["python", "/usr/local/bin/paddle-pipeline-entrypoint.py"]
