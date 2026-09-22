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

## Development

```bash
uv sync --all-extras   # install runtime + dev (pytest, ruff) deps into /home/vscode/.venv
uv run pytest          # runs tests/ (live-server tests are skipped by default)
uv run ocrbench --help # lists the prepare/infer/probe/score/calibrate/bench-latency/report stages
ocrbench prepare --config configs/run.yaml --run-id <id> [--datasets thaiocrbench]
uv run ocrbench infer --run-id <id> [--models teleocr,dotsocr]
uv run ocrbench score --run-id <id>  # scores/fields/aggregates/statements.jsonl + tob_parity.json
uv run ocrbench calibrate --run-id <id> --target-acc 0.99,0.995  # calibration.jsonl, calibration_fields.jsonl, reliability.json, thresholds.json
```

Statement-only stages (M4):

```bash
# review queue: every cross-model disagreement + a seeded 10% of agreements
uv run ocrbench review export --run-id <id>            # -> runs/<id>/review/queue.csv
# load the filled queue back as statement ground truth (rejects a bad line by number)
uv run ocrbench review load --run-id <id> --labels runs/<id>/review/queue.csv
# duplicate detection: build a separate run of originals, variants and hard negatives
uv run ocrbench dupset build --run-id <id> --dup-run-id <id>-dup
uv run ocrbench infer --run-id <id>-dup
uv run ocrbench dupset eval --dup-run-id <id>-dup      # -> runs/<id>-dup/duplicates.jsonl
```

Continuing this work on another machine (H100/L4): see
[docs/continuing-on-another-host.md](docs/continuing-on-another-host.md).

## Prepared data (DVC)

The raw statements (`data/statements`) and the prepared run `data/runs/2026-09-22-a` are
versioned with DVC on `gs://looloo-ocr-weights-and-data/vlm-ocr-bench/dvc`, so no host needs to
re-run `prepare` (about 20 minutes of CPU). They are client data: the bucket is the only copy
outside our hosts. Git holds only pointer files (`data/statements.dvc`, `dvc.yaml`, `dvc.lock`).

```bash
# once per host: point DVC at the service-account key (never commit the key or config.local)
uv run dvc remote modify --local gcs credentialpath /path/to/looloo-ocr-<id>.json
uv run dvc pull          # statements + prepared run
uv run dvc status        # "up to date" = data matches the code/configs in this commit
```

If `dvc status` reports the `prepare` stage changed, meaning the preparation code, configs or statements
changed, rebuild and publish it with `uv run dvc repro prepare && uv run dvc push`.

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
