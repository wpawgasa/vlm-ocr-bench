# Spike A: the line-reader bake-off

**Task:** docparse 1.1 (OpenSpec `add-docparse-system`). **Date:** 2026-10-08. **Host:** H100 NVL.
**Result:** muocr, read the way the checkpoint was trained (its own 64×384 resize), is the most
accurate reader on statement lines in every condition: **2.10% / 3.93% / 2.52%** mean CER
(clean / scan_low / photo) against `paddle_crop`'s **4.34% / 11.17% / 5.97%**. It reads about
230 segments per second (1.3 s per 300). `no_repeat_ngram_size` 0 is confirmed. D7's
pad-to-384 input is not: it makes muocr hallucinate into the blank margin.

The numbers come from `runs/docparse/spikes/line_readers.json`, written by
`scripts/docparse_spike_a.py report`. This note gives counts and rates only, with no statement
content.

## Line set

| Source | Lines | Notes |
|---|---|---|
| Statement text layers, run `2026-09-22-a` | 5,528 (2,110 numeric) | 27 pages: 21 KBank (4,799 lines), 2 TTB (530), 4 BBL (199) |
| ThaiOCRBench Fine-grained text recognition | 160 | the question's region; 40 multi-line regions excluded |
| ThaiOCRBench Text recognition | 0 | excluded, see below |

Each line is read under `clean`, `scan_low` and `photo`, so there are **17,064 crops**, and every
reader reads all of them (no missing line, no request error).

**Statement segments** follow design D6 on the ground-truth word boxes instead of the detector:
- words are grouped into lines;
- a line is split at gaps wider than 0.8 line heights (column breaks);
- a run is split at word boundaries so that no segment is wider than 6:1 (muocr's input);
- the margin is 0.2 line heights vertically and 0.3 horizontally, but never past half the gap to
  the next word on the line.

`photo` crops are rectified through the page's known homography, as a quad detector's crop would
be. A line is "numeric" when its text is only digits and `, . - + / :` and spaces.

**Excluded or repaired in the ground truth** (counts are in the report's `build` block):
- **Text recognition (200 samples):** these are not lines. They are whole images (up to 2048 px)
  with a question pointing at a region, without a box, and 85 have multi-line answers. A
  64×384 line reader cannot take them, and nothing in them says which line is meant.
- **Vertical margin text (92 words):** the text layer stores a rotated form code's characters in
  reverse order. These words also bridged neighbouring table rows during line grouping, so they
  are removed before grouping.
- **Detached tone marks (16 segments):** a mark extracted as its own word ends up in the wrong
  place when words are joined.
- **Two rows chained into one line (6 segments).**
- **`(cid:255)` (328 segments, all from one KBank file):** an unmapped glyph that renders as
  SARA A (U+0E30). It was checked on the crops, and every reader reads it so. It is mapped
  before scoring.
- **SARA AM:** NIKHAHIT + SARA AA is folded to SARA AM on both sides. A few text layers, and all
  of muocr's output, spell the vowel decomposed. `canonical_text` does not fold it.

The last two are also present in the main benchmark's statement ground truth (see Follow-ups).

## Readers

| Reader | How | Settings |
|---|---|---|
| `paddle_crop` | PaddleOCR-VL-1.6 on the crop, prompt `OCR:` (the pipeline's text-block read), sent straight to vLLM | |
| `typhoon_crop` | typhoon-ocr1.5-2b on the crop, its one official prompt, `typhoon_markdown` parser | |
| `dots_grounding` | dots.ocr `prompt_grounding_ocr` on the whole page plus the line's box; the reply's JSON `text` fields | smart_resize 4 MP |
| `muocr_<prep>_nr<k>` | `muocr-base-26m-stage2-finetuned-20240820-v1`, transformers 4.43.2 (the checkpoint's own), fp32 | beam 4, `max_new_tokens` 120, length penalty 2.0 |

- All VLMs ran on vLLM 0.22.1 in bf16 at T=0, with `max_tokens` 512 (a line is short; CER is capped
  at 1 anyway). Crops under 28 px are upscaled (`resize_by_need`). dots.ocr and typhoon ran on
  0.11.0 in the benchmark, so their line numbers here are not comparable latency-wise.
- muocr's `squash` input is the checkpoint's own `ViTImageProcessor`: a straight resize to 64×384.
  `pad` is design D7's: height 64 keeping the aspect ratio, padded right with the background.

## Results: mean CER per line (%)

CER is `min(1, Levenshtein / len(gt))` over `ocr_bench` `plain_text`, averaged per line.
Exact match is in brackets.

**Statement lines (5,528 per condition)**

| Reader | clean | scan_low | photo |
|---|---|---|---|
| `paddle_crop` | 4.34 (81.2) | 11.17 (69.9) | 5.97 (77.7) |
| `typhoon_crop` | 5.58 (85.4) | 6.59 (75.9) | 6.82 (77.8) |
| `dots_grounding` | 40.75 (53.3) | 41.96 (50.4) | 46.33 (46.7) |
| **`muocr_squash_nr0`** | **2.10 (93.9)** | **3.93 (91.0)** | **2.52 (93.6)** |
| `muocr_squash_nr3` | 2.11 (93.7) | 3.94 (90.8) | 2.53 (93.5) |
| `muocr_pad_nr0` | 4.31 (88.6) | 4.17 (89.4) | 4.07 (90.1) |
| `muocr_pad_nr3` | 4.32 (88.4) | 4.17 (89.3) | 4.07 (90.0) |

**Numeric statement lines (2,110 per condition)**

| Reader | clean | scan_low | photo |
|---|---|---|---|
| `paddle_crop` | **0.07 (99.9)** | **0.29 (98.4)** | **0.07 (99.6)** |
| `typhoon_crop` | 0.76 (95.8) | 1.44 (92.5) | 1.43 (92.5) |
| `dots_grounding` | 34.10 (63.5) | 32.67 (64.7) | 38.68 (57.6) |
| `muocr_squash_nr0` | 0.16 (99.8) | 0.39 (98.1) | 0.17 (99.7) |
| `muocr_squash_nr3` | 0.16 (99.8) | 0.39 (98.1) | 0.17 (99.7) |
| `muocr_pad_nr0` | 2.07 (90.5) | 0.84 (94.9) | 0.94 (94.7) |

**Text (non-numeric) statement lines (3,418 per condition)**

| Reader | clean | scan_low | photo |
|---|---|---|---|
| `paddle_crop` | 6.98 | 17.88 | 9.61 |
| `typhoon_crop` | 8.55 | 9.76 | 10.14 |
| `muocr_squash_nr0` | **3.30** | **6.12** | **3.97** |

**ThaiOCRBench Fine-grained (160 per condition, scene and document text)**

| Reader | clean | scan_low | photo |
|---|---|---|---|
| `paddle_crop` | 36.36 | 55.03 | 45.99 |
| `typhoon_crop` | **12.78** | **21.97** | **15.66** |
| `dots_grounding` | 45.23 | 50.10 | 50.54 |
| `muocr_squash_nr0` | 18.14 | 25.84 | 20.22 |

**Per bank, statement lines, clean / scan_low / photo:** KBank (4,799 lines) `paddle_crop`
4.83 / 12.56 / 6.69, muocr 2.40 / 4.47 / 2.86. BBL (199) `paddle_crop` 1.63 / 3.12 / 2.05,
muocr 0.55 / 1.34 / 1.06. TTB (530, English and numbers only) `paddle_crop` 0.94 / 1.58 / 0.84,
muocr 0.00 in every condition. The statement result is mostly a KBank result.

## Findings

1. **muocr beats `paddle_crop` on statement lines in every condition**, by 2× clean and almost
   3× on `scan_low`. `paddle_crop` stays slightly better on numeric lines (0.07% against 0.16%
   clean). muocr's errors are concentrated in very short segments: lines narrower than 2:1 have
   5.6% mean CER clean, against 0.2% for 2:1 to 4:1. One wrong character in a 3-character segment
   is 33%.
2. **Against the D7 gate, muocr narrowly misses one threshold.** Per-line mean CER is 2.10% clean
   (threshold 2%), with 3.93% / 2.52% degraded (threshold 6%), and it is no worse than
   `paddle_crop` in any condition. Pooled over characters, its clean CER is 0.85%. The gate
   (task 4.3) has to say which average it uses. This spike does not replace it: the gate runs on
   held-out lines from the frozen eval list.
3. **Keep the checkpoint's own resize, not D7's pad.** Padding to 384 costs 2.2 points clean
   (4.31% against 2.10%). On short crops, muocr writes a currency sign or `น.` into the blank
   right side (for example a bare amount read with a trailing `€` or `£`). Between 4:1 and 6:1,
   where there is little padding, pad is slightly better (0.60% against 0.74% clean).
4. **`no_repeat_ngram_size` 0 is right, but matters little here.** 39 of 17,064 lines read
   differently between 3 and 0 (squash). 0 is better on 33, 3 on 3. The typical case is a date
   range losing its last digit under 3. Only 3 differing lines are numeric: these statements
   rarely repeat a 3-token sequence in one segment. The D7 concern stands for long amounts.
5. **`paddle_crop` sometimes answers in the wrong script.** On short Thai crops it returns
   Tibetan, Telugu, Korean, Devanagari or LaTeX: 58 / 221 / 131 statement lines (clean /
   scan_low / photo) and 14–23 of the 160 Fine-grained lines. These lines carry 18–31% of its
   total CER in each slice.
6. **typhoon is the best reader of ThaiOCRBench scene text** (12.8% against muocr's 18.1%), but
   is behind both on statements. It loops on 293 of 17,064 crops (the 512-token cap) and often
   wraps a crop in `<figure>`, which the parser drops.
7. **dots.ocr grounding is not a line reader.** It reads past the box: the whole cell, the rest of
   the row or the neighbouring line, so its CER is about 40% even on clean pages. The boxes it
   echoes back match the request within a few pixels, so this is the model, not the
   coordinates. It hits the token cap on 620 crops.

## muocr throughput

H100 NVL, alone on the GPU, fp32, transformers 4.43.2, beam 4, `no_repeat_ngram_size` 0, over the
5,528 clean statement segments:

| Batch | squash: segments/s | s per 300 | pad: segments/s |
|---|---|---|---|
| 32 | 214.7 | 1.40 | 218.5 |
| 64 | 229.0 | 1.31 | 219.8 |
| 128 | 233.3 | 1.29 | 225.3 |
| 256 | 232.7 | 1.29 | 225.7 |

Preprocessing is about 3.2 s of the 24 s. This is below D11's 2–4 s per 300 segments. bf16 and
ONNX were not measured.

For comparison, throughput of the VLM readers on the same crops at 128 concurrent requests:
`paddle_crop` 64 lines/s, typhoon 124 lines/s, dots.ocr grounding 4.2 lines/s (a whole page per
line).

## Follow-ups

- **Design D7 and task 4.2:** use the checkpoint's own 64×384 resize (updated in this change).
- **Task 4.3:** define the gate's CER average (per line or pooled).
- **ocr_bench statement ground truth:** `(cid:255)` (328 lines of one KBank file) and decomposed
  SARA AM are scored as errors for every model there too. Fixing them changes published
  statement CER, so it is left to a separate change.

## Reproducing

```bash
uv run python scripts/docparse_spike_a.py build              # crops + lines.jsonl, ~2 min
scripts/serve_paddleocr_vl.sh up   # or the paddleocr_vl service alone; likewise typhoon, dotsocr
uv run python scripts/docparse_spike_a.py read --reader paddle_crop --base-url http://localhost:8005/v1 --concurrency 128
uv run python scripts/docparse_spike_a.py read --reader typhoon_crop --base-url http://localhost:8003/v1 --concurrency 128
uv run python scripts/docparse_spike_a.py read --reader dots_grounding --base-url http://localhost:8002/v1 --concurrency 128
# muocr, in the vLLM image with the checkpoint's transformers in <deps> (see the script)
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/deps \
  -v <deps>:/deps -v "$PWD:/w" -w /w --entrypoint python3 vllm/vllm-openai:v0.22.1 \
  scripts/docparse_spike_a_muocr.py
uv run python scripts/docparse_spike_a.py report             # -> runs/docparse/spikes/line_readers.json
```
