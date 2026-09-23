"""Start PaddleX's serving app for the PaddleOCR-VL-1.6 pipeline against our vLLM server.

Takes the pipeline config shipped with the installed PaddleX (/config/PaddleOCR-VL-1.6.yaml),
changes only VL recognition's `genai_config` to the documented vLLM-server backend
(`$VL_SERVER_URL`), prints the diff, and execs `paddlex --serve` on it.
"""

import os
import sys

import yaml

SRC = "/config/PaddleOCR-VL-1.6.yaml"
DST = "/config/PaddleOCR-VL-1.6.vllm-server.yaml"

with open(SRC, encoding="utf-8") as f:
    config = yaml.safe_load(f)

vl = config["SubModules"]["VLRecognition"]
before = dict(vl.get("genai_config") or {})
vl["genai_config"] = {**before, "backend": "vllm-server", "server_url": os.environ["VL_SERVER_URL"]}
print(f"VLRecognition.genai_config: {before} -> {vl['genai_config']}", flush=True)

with open(DST, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

port = os.environ.get("PORT", "8080")
argv = ["paddlex", "--serve", "--pipeline", DST, "--device", "cpu"]
argv += ["--host", "0.0.0.0", "--port", port]
print(" ".join(argv), flush=True)
sys.stdout.flush()
os.execvp(argv[0], argv)
