# Runpod worker image (design D2): one Dockerfile, one stage per variant, weights baked in.
#
#   docker build --target <plain|teleocr|paddle> -f runpod/Dockerfile \
#       --build-arg VLLM_IMAGE=<base@sha256:...> --build-arg MODEL=<name> \
#       --build-arg WEIGHTS_REPO=<hf repo> --build-arg WEIGHTS_REVISION=<40-hex commit> \
#       --secret id=HF_TOKEN,env=HF_TOKEN .
#
# VLLM_IMAGE, WEIGHTS_REPO and WEIGHTS_REVISION come from configs/runpod/<model>.yaml
# (image.base, weights.repo, weights.revision). The token is a BuildKit secret: it is
# mounted for the download step only and never lands in a layer.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.11.0

FROM ${VLLM_IMAGE} AS plain

ARG MODEL
ARG WEIGHTS_REPO
ARG WEIGHTS_REVISION

WORKDIR /app
COPY ocr_bench/__init__.py /app/ocr_bench/__init__.py
COPY ocr_bench/runpod/ /app/ocr_bench/runpod/
COPY configs/runpod/ /app/configs/runpod/
COPY configs/models/ /app/configs/models/
COPY runpod/entrypoint.py /app/entrypoint.py

# Pinned revision only: `main` or an empty value fails the build rather than baking a moving target.
RUN --mount=type=secret,id=HF_TOKEN \
    test -n "${MODEL}" && test -n "${WEIGHTS_REPO}" && test -n "${WEIGHTS_REVISION}" \
    && python3 -c "import os, sys; from huggingface_hub import snapshot_download; \
repo, rev = sys.argv[1], sys.argv[2]; \
tok = open('/run/secrets/HF_TOKEN').read().strip() if os.path.exists('/run/secrets/HF_TOKEN') else None; \
snapshot_download(repo_id=repo, revision=rev, local_dir='/models/' + repo, token=tok)" \
        "${WEIGHTS_REPO}" "${WEIGHTS_REVISION}"

ENV HF_HUB_OFFLINE=1 \
    PORT=8000 \
    OCRBENCH_MODEL=${MODEL} \
    HEALTH_CHECK_PATH=/health \
    PYTHONPATH=/app
EXPOSE 8000
ENTRYPOINT ["python3", "/app/entrypoint.py"]

# TeleOCR: the plugin and logits processor exactly as .devcontainer/teleocr-vllm.Dockerfile installs them.
FROM plain AS teleocr

ARG TELEOCR_COMMIT=9921cffe380efe4e2fa010258b3d0c3cb70bab2d

RUN pip install --no-cache-dir --no-deps \
        "TeleOCR-vllm @ git+https://github.com/caipeng328/TeleOCR@${TELEOCR_COMMIT}#subdirectory=TeleOCR-vllm" \
    && SITE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && curl -fsSL -o "${SITE}/teleocr_no_repeat_ngram.py" \
        "https://raw.githubusercontent.com/caipeng328/TeleOCR/${TELEOCR_COMMIT}/TeleOCR/vlm_utils/vlm_client/vllm_v1_no_repeat_ngram.py" \
    && python3 -c "import TeleOCR_vllm, teleocr_no_repeat_ngram; print('TeleOCR vLLM plugin + logits processor OK')"

# PaddleOCR-VL: PaddleX as .devcontainer/paddle-pipeline.Dockerfile installs it, GPU wheel.
# One worker, one public port (D3): paddle_supervisor.py runs vLLM on 127.0.0.1:8000,
# PADDLE_PIPELINES pipeline servers on 127.0.0.1:8081..., and paddle_front.py on $PORT.
FROM plain AS paddle

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# paddlepaddle-gpu 3.x is published only on Paddle's index (PyPI stops at 2.6.x); CUDA 12.6
# wheels, as the devcontainer's GPU layout path uses; the wheel brings its own CUDA libraries.
ARG PADDLE_PKG=paddlepaddle-gpu
ARG PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu126/
RUN pip install --no-cache-dir "${PADDLE_PKG}==3.3.1" -i "${PADDLE_INDEX}" \
        --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir "paddleocr[doc-parser]==3.7.0" \
    && paddlex --install serving -y \
    && paddlex --get_pipeline_config PaddleOCR-VL-1.6 --save_path /config \
    && python3 -c "import yaml; yaml.safe_load(open('/config/PaddleOCR-VL-1.6.yaml'))"

COPY .devcontainer/paddle-pipeline-entrypoint.py /app/paddle-pipeline-entrypoint.py
COPY runpod/paddle_front.py /app/paddle_front.py
COPY runpod/paddle_supervisor.py /app/paddle_supervisor.py

ENV PORT=8080 \
    PADDLE_PIPELINES=2
EXPOSE 8080
ENTRYPOINT ["python3", "/app/paddle_supervisor.py"]
