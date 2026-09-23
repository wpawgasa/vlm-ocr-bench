# Continuing on another host (H100 / L4)

State as of commit `bf8dab1` (2026-09-22). Everything that can be built and tested without a
benchmark GPU is done and on `main`: 569 tests, ruff clean. What is left needs an H100, an L4,
or a human labeller.

## 0. Status after the first H100 session (2026-09-23)

Model line-up changed: **TeleOCR is dropped** because it cannot read Thai (see below). Typhoon
OCR 1.5 is now a main model, and OvisOCR2 plus PaddleOCR-VL-1.6 are being added.

| Model | Rows in `2026-09-22-a` | State |
| --- | --- | --- |
| dots.ocr | 6,846 / 6,846, 0 errors | done (vLLM 0.11.0, bf16) |
| TeleOCR | 852 / 6,846 | stopped on purpose; kept as evidence |
| typhoon_ocr15 | 6,846 / 6,846, 0 errors | done (vLLM 0.11.0, bf16); 449 rows hit max_tokens, 457 KIE replies not JSON |
| OvisOCR2, PaddleOCR-VL-1.6 | — | integrated and unit-tested (tasks 3.10, 3.13); serve on v0.22.1 (3.11), OvisOCR2 Thai gate first (3.12) |

Serving the new models (single GPU: stop the current server first):

```bash
scripts/serve_ovisocr2.sh up        # vllm/vllm-openai:v0.22.1 (VLLM_IMAGE_NEW), --gdn-prefill-backend=triton
scripts/serve_paddleocr_vl.sh up    # PaddleOCR-VL vLLM + PaddleX pipeline server (layout on CPU)
# the dev container needs OVISOCR2_BASE_URL / PADDLEOCR_VL_BASE_URL / PADDLE_PIPELINE_URL
# (in compose.yaml; recreate it, or pass them with `docker exec -e`)
uv run ocrbench infer --run-id 2026-09-22-a --models paddleocr_vl
```

**Disk:** `vllm/vllm-openai:v0.11.0` was removed from this host on 2026-09-23 to make room for the
GPU PaddleX pipeline image. dots.ocr and typhoon ran on
`vllm/vllm-openai@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c`;
pull that digest (re-tag as `v0.11.0`) before their latency runs.

`configs/run.yaml` still lists teleocr/dotsocr: it is a `prepare` dependency, so changing it
needs `config.resolved.yaml` regenerated and `dvc commit -f prepare`, which fails while
`data/statements` is missing from the remote. Pass `--models` explicitly until then.

Findings so far (details and the evidence crop: `data/runs/2026-09-22-a/evidence/`, DVC-tracked):
- **TeleOCR does not read Thai** in bf16 with its official sampling: 0% Thai characters on
  statement pages; it reads the English and invents Korean/Chinese text instead. On a clean Thai
  header crop it drops the Thai line, with `presence_penalty` 1.0 and 0.0 alike, so it is the
  model, not the harness.
- **dots.ocr loops on Thai:** 680 rows (9.9%) hit `max_tokens` 8192 at the spec's T=0, mostly
  ThaiOCRBench repetition loops. The rerun of those rows at upstream's T=0.1 (sensitivity run
  `2026-09-22-a-dots-t01`, model config `dotsocr_t01`, never a headline number) still caps 392 of
  680, so the loops are mostly the model, not greedy decoding.

**Results are DVC-tracked per file** (`data/runs/<run>/*.dvc`; `prepare` owns only img, gt,
manifest and the two config files). With the hardlink cache these files are read-only, and
`infer` rewrites `config.infer.yaml` and appends to `predictions.jsonl` (plus a shell `>` to its
log), so **before** resuming `infer` run
`dvc unprotect data/runs/2026-09-22-a/{predictions.jsonl,config.infer.yaml,infer-<model>.log}`
(it fails with `Permission denied` otherwise), then `dvc add` + `dvc push` + commit the `.dvc`
files after it. `unprotect` makes a private copy, so predictions briefly cost ~1 GB more disk. The sensitivity run's `img`, `gt`,
`config.resolved.yaml` and `prepare_summary.json` are relative symlinks into `2026-09-22-a`
(not tracked); recreate them after a pull.

The raw statements (`data/statements`, `.dir` `c54d95d7…`) were never pushed to the DVC remote:
`dvc status` shows them deleted on every other host. Push them from the V100 box.

## 1. Set the host up

```bash
git clone https://code.loolootech.com/wichai.p/vlm-ocr-bench.git && cd vlm-ocr-bench
# Reopen in the dev container (VS Code), or: devcontainer up --workspace-folder .
uv sync --all-extras
uv run python -m nltk.downloader -q wordnet omw-1.4     # NLTK METEOR needs WordNet
```

Per-host settings go in `.devcontainer/.env` (gitignored, see `.env.example`): `DATA_DIR`,
`MODELS_DIR`, `HF_TOKEN`, GPU ids, ports, `VLLM_IMAGE`.

**Data** (client bank statements: keep them on our hosts only):

```bash
cp /secure/path/looloo-ocr-<id>.json .            # the DVC service-account key, gitignored
uv run dvc remote modify --local gcs credentialpath "$PWD/looloo-ocr-<id>.json"
uv run dvc pull        # ~7.3 GB: data/statements + the prepared run data/runs/2026-09-22-a
uv run dvc status      # expect "Data and pipelines are up to date" — do NOT re-run prepare
```

The prepared run is **`2026-09-22-a`**: 2,282 samples and 6,846 manifest rows (1,824
ThaiOCRBench + 458 statement pages, each under `clean`, `scan_low` and `photo`).

## 2. Start the model servers (H100, bf16)

```bash
scripts/serve_teleocr.sh up     # builds ocr-bench/teleocr-vllm:9921cff on first use (~35 min)
scripts/serve_dotsocr.sh up
```

- TeleOCR needs **its own image**: `.devcontainer/teleocr-vllm.Dockerfile` adds the
  `TeleOCR_vllm` plugin plus the `no_repeat_ngram_size` V1 logits processor, both pinned to
  upstream commit `9921cff`. Compose passes `--logits-processors`, and the harness sends
  `extra_body={"vllm_xargs": {"no_repeat_ngram_size": 100}}`.
- Do **not** use `COMPOSE_EXTRA=.devcontainer/compose.v100.yaml` on the H100. That override is
  fp16 plus sm_70 workarounds, and its numbers must never be quoted.
- Optional third reference model: `docker compose -f .devcontainer/compose.yaml --profile typhoon up -d`
  and add `typhoon_ocr15` to `configs/run.yaml`.

## 3. The work that is left

### 3.9 (issue #3) — full inference, the gate for everything below
```bash
uv run ocrbench infer --run-id 2026-09-22-a --models teleocr,dotsocr
uv run ocrbench probe --run-id 2026-09-22-a
```
- `infer` is resumable: rows already present without an error are skipped, so an interrupted run
  can simply be re-run. It refuses to finish if row counts don't match the manifest.
- Accept only an error rate under 2%, and check that the per-model summary shows no silent drops.
- This also completes **3.8**, whose dots.ocr live path could not be verified on the V100: vLLM
  0.11 on sm_70 falls back to FlexAttention, which crashed with a CUDA illegal memory access, and
  is far too slow (about 10 s for a 9-token reply). The V100 override already disables the
  FlashInfer sampler, which is a separate sm_70 crash.
- **Watch for:** TeleOCR rendered Thai as romanisation, Korean or Chinese in the V100 fp16 smoke
  test, while reading English perfectly. If bf16 shows the same, that is a real finding about the
  model (it is trained for Chinese and English), not a harness bug — a `probe.json` and a few
  sample outputs make the case for the client deck.

### Then, in order
```bash
uv run ocrbench score --run-id 2026-09-22-a          # ~30-40 s of that is the tob parity gate
uv run ocrbench review export --run-id 2026-09-22-a  # -> runs/2026-09-22-a/review/queue.csv
#   label 30-50 pages in the CSV's empty `label` column (task 5.8, ~4 h of human work)
uv run ocrbench review load --run-id 2026-09-22-a --labels <filled.csv>
uv run ocrbench score --run-id 2026-09-22-a          # re-score with the manual ground truth
uv run ocrbench calibrate --run-id 2026-09-22-a --target-acc 0.99,0.995   # task 6.5
```
- `score` writes `scores.jsonl`, `fields.jsonl`, `aggregates.jsonl`, `statements.jsonl` and
  `tob_parity.json`. Check `tob_parity.json` says every task is comparable; if a task is not, the
  report must mark its official score "not comparable" rather than print it beside published
  numbers.
- `calibrate` reports ECE and the review rate per model, which is the Q3 answer. With too few
  labelled fields it records `insufficient_data` instead of fitting, so the manual set matters.

### Duplicates (issue #5, task 5.7 evaluation half)
```bash
uv run ocrbench dupset build --run-id 2026-09-22-a --dup-run-id 2026-09-22-a-dup
uv run ocrbench infer --run-id 2026-09-22-a-dup --models teleocr,dotsocr
uv run ocrbench dupset eval --dup-run-id 2026-09-22-a-dup
```
Hard negatives are same-bank, not same-account: no account number is known before extraction.

### Issues #7 (latency) and #8 (report) are not built yet
`bench-latency` and `report` are still stubs, exiting 3 with the task they need. #7 needs the
H100 **and** an L4 session with the same vLLM version and dtype, and #8 needs a completed run.

## 4. Things that will trip you up

- **Never quote V100 numbers.** fp16 on sm_70 is smoke-testing only.
- **`pythainlp` is pinned `<5.2`.** 5.2 changed newmm's dictionary and breaks exact parity with
  the published ThaiOCRBench scores. Do not relax the pin without re-running
  `tests/test_tob_parity.py`, which must stay at 0.01.
- **DVC staleness:** the `prepare` stage depends on `configs/run.yaml`, `configs/datasets`,
  `data/statements` and `ocr_bench/data`. Touching those marks the 7 GB run stale. If the change
  cannot affect prepared output (for example a scoring-only key in `bankstmt.yaml`), regenerate
  just `config.resolved.yaml` and `dvc commit -f prepare`; otherwise `dvc repro prepare`
  (about 19 min) and `dvc push`. Always `dvc push` before committing a changed `dvc.lock`.
- **Client data hygiene:** `data/`, `runs/` and `review/` stay gitignored; only the anonymised
  samples in the report are client-facing. Never put real statement content in tests or fixtures.
  Audit staged files for the key, `.dvc/config.local` and PDFs before each commit.
- **CI is GitHub Actions** (`.github/workflows/ci.yml`: ruff + `pytest -m "not live"` in the
  `uv:python3.11-bookworm-slim` container, on push/PR to `main`) on
  https://github.com/wpawgasa/vlm-ocr-bench. It replaced the Drone pipeline.
- **Statement quirks already handled** (see `ocr_bench/normalize/statement.py` and
  `metrics/statement_gt.py`): KBank's combined withdrawal/deposit column is split from the data,
  not the header; TTB prints newest-first and signed amounts; BBL opens with a `B/F` row; KTB and
  Krungsri PDFs have broken font encodings, so their text layers are unusable and those pages
  carry no ground truth (27 usable digital pages in total).

## 5. Planning artifacts

`openspec/changes/add-ocr-benchmark-harness/` holds the proposal, design, per-capability specs and
`tasks.md`, which is the authoritative checklist. Gitea issues #1–#8 mirror its task groups; tick
both when a task is finished. `openspec validate add-ocr-benchmark-harness --strict` must pass.
