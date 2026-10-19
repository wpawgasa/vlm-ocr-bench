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

## Client sample: document to Markdown (any model)

`scripts/runpod_parse_document.py` is a standalone client. It imports nothing from this repo
and needs `httpx`, `pillow` and `pypdfium2`, plus `opencv-python-headless` and `numpy` for
TeleOCR polygon blocks. It sends a PDF or images to any of the five endpoints and writes
Markdown:

```bash
export TYPHOON_BASE_URL=https://<endpoint-id>.api.runpod.ai/v1
export TYPHOON_API_KEY=<the restricted client key: Serverless Read/Write, no management access>   # falls back to $RUNPOD_API_KEY
python scripts/runpod_parse_document.py --model typhoon_ocr15 statement.pdf -o statement.md
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
