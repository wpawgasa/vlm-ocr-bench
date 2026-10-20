# Runpod OCR client

`runpod_parse_document.py` sends a document (PDF or images) to one of our OCR models on
Runpod and writes the text as Markdown. It is a single file with three dependencies.

## 1. Requirements

- Python **3.10 or newer**
- Network access to `https://*.api.runpod.ai`
- From us: the **endpoint URL** of each model you will use and the **client API key**
  (sent separately; never commit them)

## 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# only if you will use --model teleocr:
pip install -r requirements-teleocr.txt
```

## 3. Configure

Each model has its own pair of environment variables:

| `--model` | URL variable | Key variable |
|---|---|---|
| `dotsocr` | `DOTSOCR_BASE_URL` | `DOTSOCR_API_KEY` |
| `typhoon_ocr15` | `TYPHOON_BASE_URL` | `TYPHOON_API_KEY` |
| `paddleocr_vl` | `PADDLEOCR_VL_BASE_URL` | `PADDLEOCR_VL_API_KEY` |
| `teleocr` | `TELEOCR_BASE_URL` | `TELEOCR_API_KEY` |
| `ovisocr2` | `OVISOCR2_BASE_URL` | `OVISOCR2_API_KEY` |

The URL has the form `https://<endpoint-id>.api.runpod.ai/v1`. The client key is the same
for every model. If a model's key variable is not set, `RUNPOD_API_KEY` is used.

```bash
export DOTSOCR_BASE_URL=https://<endpoint-id>.api.runpod.ai/v1
export DOTSOCR_API_KEY=<client key>
```

You can also pass `--base-url` and `--api-key` on the command line. Variables keep the key out
of your shell history.

**Which model?** `dotsocr`, `typhoon_ocr15` and `paddleocr_vl` are the models deployed now,
and all three read Thai. `teleocr` and `ovisocr2` are supported by the script but are not
deployed.

## 4. Run

```bash
# a PDF; every page is rendered at 200 dpi
python runpod_parse_document.py --model dotsocr statement.pdf -o statement.md

# images, in page order; multi-page TIFF works too
python runpod_parse_document.py --model typhoon_ocr15 page1.png page2.jpg -o pages.md

# Markdown to stdout, without page headers, footers or page numbers
python runpod_parse_document.py --model paddleocr_vl scan.pdf --drop-page-furniture
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--model` | (required) | `dotsocr`, `typhoon_ocr15`, `paddleocr_vl`, `teleocr`, `ovisocr2` |
| `-o, --output` | stdout | Markdown file to write |
| `--concurrency` | 4 | pages sent at the same time |
| `--wait-s` | 600 | how long to wait for a worker to start |
| `--drop-page-furniture` | off | leave out page headers, footers and page numbers |
| `--base-url`, `--api-key` | env | override the variables above |

## 5. Output

The output is one Markdown document, with one section per page:

```markdown
<!-- page 1 -->

# Account statement

| Date | Description | Amount |
| --- | --- | --- |
| 01/10/2026 | Transfer | 1,000.00 |

<!-- page 2 -->
...
```

- Tables become pipe tables. A table with merged cells stays HTML (`<table>…</table>`),
  which Markdown viewers render.
- Formulas are LaTeX between `$$`. Pictures are `<!-- picture -->`. Typhoon describes
  figures in Thai; the description is a `> **Figure:** …` quote.
- When a model's answer can't be fully read, the page keeps the text that could be read
  and adds `<!-- parse error: … -->`. Check those pages by hand.

## 6. What to expect

- **Cold start.** The endpoints scale to zero when idle, after 2 minutes with no requests.
  The first request after that waits for a worker: usually 1.5 to 2 minutes, and up to about
  12 minutes on a host that has never pulled the model image. The script waits on its own
  and then sends the pages. Raise `--wait-s` if you see "no worker ready".
- **Speed.** A page takes about 3 to 20 seconds once a worker is up, depending on the model
  and the page.
- **Limits.** Each request may run up to 5.5 minutes and carry up to 30 MB, which is far more
  than a page needs. Each endpoint runs at most 2 workers; keep `--concurrency` at 8 or
  below. Above that, requests get dropped and retried.
- **Retries.** Rate limits (429), gateway errors (5xx) and timeouts are retried 3 times with
  backoff.

## 7. Errors

| Exit code | Meaning | What to do |
|---|---|---|
| 0 | done | |
| 2 | bad input or configuration (missing variable, file not a PDF or image) | fix the arguments or the variables |
| 3 | no worker became ready within `--wait-s` | retry later, or raise `--wait-s`; the GPU pool may be busy |
| 4 | the endpoint answered with an HTTP error | 401/403: check the key; other codes: retry, then contact us |

## 8. Data handling

- Pages are rendered and sent in memory. The script writes only the Markdown you ask for.
- The script requires `https://` for Runpod endpoints.
- Keep the key in an environment variable or a secret store. Don't put it in code, tickets
  or chat.
