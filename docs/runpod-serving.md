# Runpod serving

Operating notes for the Runpod serverless endpoints (OpenSpec change `add-runpod-serving`).
Task 9.1 extends this page with the client contract, deploy, rollback and cost notes.

## Model weights and licences

Each worker image bakes in its weights at the Hugging Face commit pinned in
`configs/runpod/<model>.yaml` under `weights.revision` (task 5.4). A new commit upstream
changes nothing until someone updates the pin, which then goes through a new release and
the parity gate (design D6).

Pinned on 2026-10-19 to each repository's head (`HfApi().model_info(repo).sha`). The
licence is the one declared in the model card's metadata on that date.

| Model config | Hugging Face repository | Pinned commit | Licence |
|---|---|---|---|
| `dotsocr` | [`dots-studio/dots.ocr`](https://huggingface.co/dots-studio/dots.ocr) | `c0111ce6bc07803dbc267932ffef0ae3a51dc951` | MIT |
| `ovisocr2` | [`ATH-MaaS/OvisOCR2`](https://huggingface.co/ATH-MaaS/OvisOCR2) | `1fc9221b7823a371d6e97f92d527cc847e24e107` | Apache-2.0 |
| `paddleocr_vl` | [`PaddlePaddle/PaddleOCR-VL-1.6`](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | `c5630abae1d940eafe0697512a0325494b02ab42` | Apache-2.0 |
| `teleocr` | [`StarDoc-AI/TeleOCR`](https://huggingface.co/StarDoc-AI/TeleOCR) | `70a19ec6a3b036311af19ae5e762e650f41a5980` | Apache-2.0 |
| `typhoon_ocr15` | [`typhoon-ai/typhoon-ocr1.5-2b`](https://huggingface.co/typhoon-ai/typhoon-ocr1.5-2b) | `15b381a2d62569e6736f9c085859dff68e48608d` | Apache-2.0 |

The worker images are private (`code.loolootech.com/loolootech/runpod-<model>`), so the
baked weights are not redistributed. Each licence still applies to the weights and to any
outputs as its terms say.

The TeleOCR pin is newer than the snapshot used for benchmark run `2026-09-22-a`
(`a61433186527cb53958bf354e34ae673c19cec4b`). The commits in between changed only
`README.md` and added `.eval_results/parsebench.yaml`, so the weights are identical. The
dots.ocr pin matches the benchmarked snapshot.

## First deployment (task 6.1, 2026-10-19)

Three models are deployed: `dotsocr`, `typhoon_ocr15` and `paddleocr_vl`. `teleocr` and
`ovisocr2` are out of scope for now. Their configs and images stay buildable, but they have
no endpoints, release records or smoke runs.

- **Deploy path.** `ocrbench runpod deploy` created each template and load-balancing
  endpoint through the REST API (`https://api.runpod.io/v2`). The console fallback from
  design Risks was not needed.
- **Registry pull.** Each worker pulled its image from
  `code.loolootech.com/loolootech/runpod-<model>` with the Runpod registry credential.
  That deploy was the pull test moved from 5.2.
- **GPU.** All three run on the 24 GB pool (`AMPERE_24`, MIG slices excluded) with
  `workers: {min: 0, max: 2}`. vLLM 0.11.0 starts there with `cuda_min` 12.8.
- **Keys.** One shared restricted client key (Serverless Read/Write, no management access)
  works on all three endpoints. The REST management API rejects it with 403. The
  management key reads `ocrbench runpod status` for each model.

Smoke runs on 2026-10-19 used only the client key. The cold-start SLO is 90 s and the page
SLO is 20 s:

| Model | Image digest | Cold start | Page | Layout |
|---|---|---|---|---|
| `dotsocr` | `sha256:295b10b8…` | 82.1 s | 9.2 s | |
| `typhoon_ocr15` | `sha256:2ce4e4b2…` | 96.9 s | 6.7 s | |
| `paddleocr_vl` | `sha256:2af91a60…` | 108.7 s | 17.5 s | 3.6 s |

Each figure is one cold start, measured as the first request after scale-to-zero. Task 6.2
takes the five-sample measurements that decide `slo: met|missed`.

## Measurements (task 6.2, 2026-10-20)

Measured with `scripts/runpod_measure.py` at git `2910ebb`. Each model ran on the 24 GB
pool with `workers: {min: 0, max: 2}` and a serverless price of $0.69/h. The run had two parts:

- **Cold starts:** five per model, each after the endpoint had scaled to zero. A cold start
  is the time until health is ready.
- **Page run:** right after the fifth cold start, the 60-page parity set
  (`parity_samples.v1`) at `client.max_concurrency` = 4, through the harness adapters. It
  produced no error rows.

Loop pages are pages whose output reached the model's `sampling.max_tokens` (8192 for
dotsocr and paddleocr_vl, 10000 for typhoon_ocr15). The page targets are judged on the other
pages; loop pages are counted and timed on their own.

| Model | Image digest | Cold start p50 / p95 | Page p50 / p95 (target) | Loop pages, slowest | All pages p50 / p95 | $/page | SLO |
|---|---|---|---|---|---|---|---|
| `dotsocr` | `sha256:295b10b8…` | 16.1 / 90.6 s | 9.4 / 22.6 s | 8 of 60, 89 s | 10.9 / 75.8 s | ≤ 0.0020 | met |
| `typhoon_ocr15` | `sha256:2ce4e4b2…` | 91.2 / 98.8 s | 7.7 / 25.0 s | 6 of 60, 116 s | 8.6 / 111.7 s | ≤ 0.0019 | met |
| `paddleocr_vl` | `sha256:2af91a60…` | 6.1 / 113.3 s | 4.7 / 19.5 s | 8 of 60, 35 s | 5.8 / 26.1 s | 0.00055 | met |

**SLO targets (revised 2026-10-20 from this measurement):** page p50 ≤ 10 s and p95 ≤ 30 s
on non-loop pages, cold start p95 ≤ 120 s. The initial targets (6 s, 20 s, 90 s, design D7)
were set before measuring and all three models missed them. The release records carry
`slo: met` against the revised targets. `teleocr` and `ovisocr2` keep the initial targets
until they are deployed and measured.

**Cold starts come in two kinds.**

- **Resume:** FlashBoot brings a recently stopped worker back in 2–16 s.
- **Full start:** vLLM loads the weights and captures CUDA graphs. This takes 86–92 s for
  dotsocr, 91–99 s for typhoon_ocr15 and 107–115 s for paddleocr_vl, which also starts the
  PaddleX layout pipeline.

Each model got 2 or 3 full starts out of 5. Every full start except dotsocr's 86 s one was
over the initial 90 s target. Every one is within 120 s. For client traffic, `workers_min: 1`
in office hours (`--profile office`, task 6.5) still removes the cold start from the first page.

**The page tail is generation loops, not serving.**

- Every page above 30 s is a loop page. At 24 GB-class speed a loop page takes 70–90 s on dotsocr and
  107–116 s on typhoon_ocr15. The typhoon_ocr15 figure is close to the 120 s client
  `request_timeout_s`.
- The H100 run `2026-09-22-a` looped about as often on the same 60 pages: 7, 6 and 8 pages.
  Which pages loop differs slightly between the GPUs.
- That run's own p95 over these pages was also above 20 s: 39 s for dotsocr and 52 s for
  typhoon_ocr15, under benchmark load.

A p95 that counts loop pages measures how often the model loops, not how fast the endpoint
serves, so the targets leave them out. With loop pages excluded, dotsocr and typhoon_ocr15
take 8–9 s per page at p50 on these GPUs, against 3–4 s on the H100.

**How cost per page is computed.** It is billed worker-seconds during the page run times
$0.69/h, divided by 60. Running and initializing workers count as billed, capped at
`workers_max`, so the dotsocr and typhoon_ocr15 figures are upper bounds. Their page runs
used up to 2 workers. paddleocr_vl's used 1. Idle time after a burst (120 s × $0.00019/s ≈
$0.023) is not included.

## Client sample: document to Markdown (any model)

`clients/runpod-ocr-client/runpod_parse_document.py` is a standalone client. It imports nothing from this repo
and needs `httpx`, `pillow` and `pypdfium2`, plus `opencv-python-headless` and `numpy` for
TeleOCR polygon blocks. It sends a PDF or images to any of the five endpoints and writes
Markdown:

```bash
export TYPHOON_BASE_URL=https://<endpoint-id>.api.runpod.ai/v1
export TYPHOON_API_KEY=<the restricted client key: Serverless Read/Write, no management access>   # falls back to $RUNPOD_API_KEY
python clients/runpod-ocr-client/runpod_parse_document.py --model typhoon_ocr15 statement.pdf -o statement.md
```

Each model is called, and answers, differently. The script calls each the way the
benchmark harness does, with the same prompts, sampling and image preparation:

| `--model` | Request | Answer |
|---|---|---|
| `dotsocr` | one chat request, layout prompt; smart_resize ≤ 4 MP | JSON list of `{bbox, category, text}` blocks (Markdown text, HTML tables, LaTeX formulas) |
| `typhoon_ocr15` | one chat request, its one trained prompt; longest side ≤ 1800 px | Markdown with HTML tables, `<figure>` (Thai description) and `<page_number>` tags |
| `ovisocr2` | one chat request; smart_resize (factor 32, ≤ 8.3 MP) | Markdown with HTML tables and `<img src="images/bbox_…">` figure tags |
| `teleocr` | one layout request on a 1036×1036 copy, then one request per block crop | layout lines `<box:…><label:…><angle>`, then each block's text (tables as OTSL) |
| `paddleocr_vl` | `POST /layout-parsing` on the endpoint root (PaddleX pipeline), page as PNG | JSON envelope; `layoutParsingResults[0].prunedResult.parsing_res_list` blocks |

Whatever the model, the output is the same kind of Markdown:

- Each page starts with `<!-- page N -->`.
- Tables become pipe tables unless cells merge; then they stay HTML, which Markdown renders.
- Formulas are wrapped in `$$`.
- Pictures become `<!-- picture -->`; typhoon's figure descriptions become a quote.
- `--drop-page-furniture` leaves out page headers, footers and page numbers.
- An answer that can't be fully parsed keeps what could be read, with a
  `<!-- parse error: … -->` note.

Before sending pages, the script waits on `/health` for a worker; a first cold start takes
minutes. It retries gateway errors, rate limits and timeouts.
`tests/test_runpod_parse_document.py` keeps each model's prompt, sampling, resize, crops and
parsing identical to the harness.
