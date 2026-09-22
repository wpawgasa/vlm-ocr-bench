# vLLM server image for TeleOCR (design D14).
#
# TeleOCR's official vLLM path needs two additions to the stock image:
#   1. the TeleOCR_vllm plugin, which registers its modified Qwen2.5-VL class
#      (auto-loaded through the `vllm.general_plugins` entry point);
#   2. its V1 logits processor implementing `no_repeat_ngram_size`, loaded with
#      `--logits-processors teleocr_no_repeat_ngram:VllmV1NoRepeatNGramLogitsProcessor`
#      and driven per request by `extra_body={"vllm_xargs": {"no_repeat_ngram_size": N}}`.
# Both are pinned to one upstream commit. --no-deps keeps the image's vLLM/transformers,
# which already match the plugin's pins (vllm==0.11.0).
ARG VLLM_IMAGE=vllm/vllm-openai:v0.11.0
FROM ${VLLM_IMAGE}

ARG TELEOCR_COMMIT=9921cffe380efe4e2fa010258b3d0c3cb70bab2d

RUN pip install --no-cache-dir --no-deps \
        "TeleOCR-vllm @ git+https://github.com/caipeng328/TeleOCR@${TELEOCR_COMMIT}#subdirectory=TeleOCR-vllm" \
    && SITE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && curl -fsSL -o "${SITE}/teleocr_no_repeat_ngram.py" \
        "https://raw.githubusercontent.com/caipeng328/TeleOCR/${TELEOCR_COMMIT}/TeleOCR/vlm_utils/vlm_client/vllm_v1_no_repeat_ngram.py" \
    && python3 -c "import TeleOCR_vllm, teleocr_no_repeat_ngram; print('TeleOCR vLLM plugin + logits processor OK')"
