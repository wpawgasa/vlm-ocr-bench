# ocr-bench — TeleOCR vs dots.ocr for บสย. / LooLoo

Spec: [docs/OCR Benchmark Spec — TeleOCR vs dots.ocr for บสย. LooLoo questions.md](<docs/OCR Benchmark Spec — TeleOCR vs dots.ocr for บสย. LooLoo questions.md>)

## Dev container

The harness never loads model weights; it talks to vLLM OpenAI-compatible servers (spec §4).
The dev container therefore has no CUDA — Python 3.11, `uv`, poppler (`pdffonts`/`pdftoppm`/`pdftotext`),
Thai fonts and opencv runtime libs. Code is written on the V100 dev box; benchmark runs happen on H100.

Open the folder in VS Code → **Reopen in Container**, or with the CLI:

```bash
DOCKER_BUILDKIT=1 devcontainer up --workspace-folder .
```

`uv sync --all-extras` runs automatically once `pyproject.toml` exists. The venv is at `/home/vscode/.venv`.

| Host | Container | Purpose |
| --- | --- | --- |
| repo | `/workspaces/vlm-ocr-bench` | source |
| `DATA_DIR` (default `./data`, gitignored) | `/data` (`$OCRBENCH_DATA_DIR`) | ThaiOCRBench exports, bank statements, runs |
| `MODELS_DIR` (default `~/.cache/huggingface`) | `~/.cache/huggingface` in dev, `/root/.cache/huggingface` in vLLM | model weights, downloaded once |

Per-host overrides (paths, GPU ids, ports, vLLM image tag, `HF_TOKEN`) go in `.devcontainer/.env`,
see `.devcontainer/.env.example`.

## Model servers

Defined in `.devcontainer/compose.yaml` under the `serve` profile (bf16, spec §4.3 flags):

| Service | Model | Host port | In-container URL |
| --- | --- | --- | --- |
| `teleocr` | `StarDoc-AI/TeleOCR` | 8001 | `$TELEOCR_BASE_URL` = `http://teleocr:8000/v1` |
| `dotsocr` | `dots-studio/dots.ocr` | 8002 | `$DOTSOCR_BASE_URL` = `http://dotsocr:8000/v1` |
| `typhoon` (profile `typhoon`, optional) | `typhoon-ai/typhoon-ocr1.5-2b` | 8003 | `$TYPHOON_BASE_URL` |

```bash
scripts/serve_teleocr.sh up     # also: down | logs
scripts/serve_dotsocr.sh up
```

The scripts work from the host or from inside the dev container. On the V100 dev box, smoke tests only:
`COMPOSE_EXTRA=.devcontainer/compose.v100.yaml scripts/serve_dotsocr.sh up` (fp16 plus the old-driver workaround). Don't quote those numbers.
