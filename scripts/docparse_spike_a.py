"""docparse Spike A (task 1.1): the line-reader bake-off.

Builds one set of line crops and reads it with every candidate line reader, so every reader is
scored on the same lines (the gate's relative condition, design D7):

    build   statement text-layer lines + single-line ThaiOCRBench Fine-grained regions, under
            clean, scan_low and photo, as crops in <work>/crops/ and <work>/lines.jsonl
    read    one vLLM-served reader over every line (resumable):
              paddle_crop     PaddleOCR-VL-1.6 "OCR:" on the crop (the pipeline's text-block read)
              typhoon_crop    typhoon-ocr1.5-2b, its one official prompt, on the crop
              dots_grounding  dots.ocr grounding OCR: the whole page plus the line's box
    report  CER per reader x source x condition, numeric lines apart -> line_readers.json

muocr runs in-process on the GPU in `scripts/docparse_spike_a_muocr.py` and writes its
readings to the same work dir.

Statement segments follow design D6 on ground-truth word boxes instead of the detector:
text-layer words grouped into lines, split at column-sized gaps, then at word boundaries so
no segment is wider than `MAX_ASPECT` (muocr's 64x384). Photo crops are rectified through the
known `photo` homography, as a quad detector's crop would be. Readings hold client text and
stay under runs/ (gitignored); the report holds numbers only.
"""

import argparse
import asyncio
import json
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

from ocr_bench.config import Sampling, load_model_config
from ocr_bench.data.degrade import degrade
from ocr_bench.metrics.cer_wer import cer
from ocr_bench.metrics.statement_gt import group_lines
from ocr_bench.models.base import parse_region
from ocr_bench.models.imaging import region_to_pixels, resize_by_need
from ocr_bench.models.parsers import ParseContext, parse
from ocr_bench.models.registry import build_model
from ocr_bench.normalize.text import plain_text
from ocr_bench.schemas import Condition, TextLayerGT

RUN_DIR = Path("data/runs/2026-09-22-a")
WORK = Path("runs/docparse/spikes/line_a")
REPORT = Path("runs/docparse/spikes/line_readers.json")

MAX_ASPECT = 6.0  # muocr's 64x384 input (design D6)
GAP_FACTOR = 0.8  # a gap wider than 0.8 line heights starts a new segment (a column break)
PAD_Y, PAD_X = 0.2, 0.3  # crop margin as a fraction of the line height
CONDITIONS = ["clean", "scan_low", "photo"]
NUMERIC_RE = re.compile(r"^[0-9,.\-+/: ]*[0-9][0-9,.\-+/: ]*$")
# KBank's text layer leaves one glyph unmapped as "(cid:255)"; it renders as SARA A
# (checked on the crops, and every reader reads it so).
CID_MAP = {"(cid:255)": "\u0e30"}
LINE_MAX_TOKENS = 512  # a line is short; caps repetition loops (CER is capped at 1 anyway)

READERS = {
    "paddle_crop": {"model": "paddleocr_vl", "prompt": "OCR:", "parser": "markdown"},
    "typhoon_crop": {"model": "typhoon_ocr15", "prompt": None, "parser": "typhoon_markdown"},
    "dots_grounding": {
        "model": "dotsocr",
        "prompt": "Extract text from the given bounding box on the image "
        "(format: [x1, y1, x2, y2]).\nBounding Box:\n",
        "parser": "markdown",
    },
}


# --- build ---------------------------------------------------------------------------------


def _manifest() -> list[dict]:
    with open(RUN_DIR / "manifest.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _segments(words) -> list[tuple[list, tuple[float, float, float, float]]]:
    """Ground-truth word boxes -> line segments (words, padded box), design D6.

    The horizontal margin never crosses half the gap to the next word on the same line, so a
    segment split off a run of close words does not take in its neighbour's glyphs."""
    out = []
    for line in group_lines(words):
        h = max(line.height, 1.0)
        runs: list[list] = [[line.words[0]]]
        for prev, word in zip(line.words, line.words[1:], strict=False):
            if word.bbox[0] - prev.bbox[2] > GAP_FACTOR * h:
                runs.append([word])
            else:
                runs[-1].append(word)
        chunks: list[list] = []
        for run in runs:
            chunk: list = []
            for word in run:
                trial = [*chunk, word]
                width = trial[-1].bbox[2] - trial[0].bbox[0] + 2 * PAD_X * h
                if chunk and width / (h * (1 + 2 * PAD_Y)) > MAX_ASPECT:
                    chunks.append(chunk)
                    chunk = [word]
                else:
                    chunk = trial
            if chunk:
                chunks.append(chunk)
        order = line.words
        for chunk in chunks:
            first, last = order.index(chunk[0]), order.index(chunk[-1])
            x0 = min(w.bbox[0] for w in chunk)
            x1 = max(w.bbox[2] for w in chunk)
            left = PAD_X * h
            if first > 0:
                left = max(0.0, min(left, (x0 - order[first - 1].bbox[2]) / 2))
            right = PAD_X * h
            if last + 1 < len(order):
                right = max(0.0, min(right, (order[last + 1].bbox[0] - x1) / 2))
            y0 = min(w.bbox[1] for w in chunk) - PAD_Y * h
            y1 = max(w.bbox[3] for w in chunk) + PAD_Y * h
            out.append((chunk, (x0 - left, y0, x1 + right, y1)))
    return out


def _vertical_zones(words) -> list[tuple[float, float, float, float]]:
    """Boxes around rotated (vertical) text: words of 4+ characters that are taller than
    wide, widened to catch the column's shorter pieces. Their text-layer characters come out
    reversed, and docparse never hands a horizontal line reader a vertical column."""
    zones = []
    for w in words:
        x0, y0, x1, y1 = w.bbox
        if y1 - y0 > 1.2 * (x1 - x0) and len(plain_text(w.text)) >= 4:
            m = 2 * (x1 - x0)
            zones.append((x0 - m, y0 - 3 * m, x1 + m, y1 + 3 * m))
    return zones


def _in_zones(box, zones) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(z[0] <= cx <= z[2] and z[1] <= cy <= z[3] for z in zones)


def _clip(box, size: tuple[int, int]) -> tuple[float, float, float, float]:
    w, h = size
    x0, y0, x1, y1 = box
    return (max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1))


def _crop(img: np.ndarray, box, hmat: np.ndarray | None) -> tuple[np.ndarray, list[float]]:
    """The crop of clean-page `box` from `img`, and its axis-aligned box in `img`.
    With a homography, the box's quad is mapped and rectified back to the box's size."""
    x0, y0, x1, y1 = box
    if hmat is None:
        r = [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]
        return img[r[1] : r[3], r[0] : r[2]], [float(v) for v in r]
    quad = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(quad, hmat).reshape(4, 2).astype(np.float32)
    bw, bh = max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))
    dst = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)
    warp = cv2.getPerspectiveTransform(mapped, dst)
    out = cv2.warpPerspective(
        img, warp, (bw, bh), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )
    hh, ww = img.shape[:2]
    hull = [
        float(np.clip(mapped[:, 0].min(), 0, ww)),
        float(np.clip(mapped[:, 1].min(), 0, hh)),
        float(np.clip(mapped[:, 0].max(), 0, ww)),
        float(np.clip(mapped[:, 1].max(), 0, hh)),
    ]
    return out, hull


def _record(lid, source, sample_id, bank, condition, gt, crop, page_rel, box, size, work):
    rel = f"crops/{lid.replace(':', '_')}.png"
    cv2.imwrite(str(work / rel), crop)
    text = plain_text(gt)
    return {
        "id": lid,
        "source": source,
        "sample_id": sample_id,
        "bank": bank,
        "condition": condition,
        "gt": gt,
        "numeric": bool(NUMERIC_RE.match(text)),
        "crop": rel,
        "page": page_rel,
        "page_size": list(size),
        "box": [round(v, 1) for v in box],
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
    }


def cmd_build(args) -> None:
    work = Path(args.work)
    (work / "crops").mkdir(parents=True, exist_ok=True)
    rows = _manifest()
    lines: list[dict] = []
    stats: dict[str, int] = defaultdict(int)

    # Statement text-layer pages (gt_kind text_layer): segments from the clean page's words.
    statement = sorted(
        {r["sample_id"] for r in rows if r["source"] == "bankstmt" and r["gt_kind"] == "text_layer"}
    )
    for sid in statement:
        by_cond = {r["condition"]: r for r in rows if r["sample_id"] == sid}
        gt = TextLayerGT.model_validate_json((RUN_DIR / by_cond["clean"]["gt_path"]).read_text())
        clean = _bgr(RUN_DIR / by_cond["clean"]["image_path"])
        size = (clean.shape[1], clean.shape[0])
        _, hmat = degrade(clean, sid, Condition.photo)
        stats["statement_pages"] += 1
        # Vertical words go before line grouping: they would bridge and merge table rows.
        zones = _vertical_zones(gt.gt_words)
        kept = [w for w in gt.gt_words if not _in_zones(w.bbox, zones)]
        stats["statement_vertical_words_excluded"] += len(gt.gt_words) - len(kept)
        for i, (words, raw_box) in enumerate(_segments(kept)):
            gt_text = " ".join(w.text for w in words)
            if not plain_text(gt_text):
                stats["statement_empty_segments"] += 1
                continue
            # Text-layer artefacts the crop cannot match: a tone mark extracted as its own
            # word (joined with a space, in the wrong place), and words of two rows chained
            # into one line.
            if any(
                all(unicodedata.category(c) == "Mn" for c in plain_text(w.text) or "x")
                for w in words
            ):
                stats["statement_detached_mark_excluded"] += 1
                continue
            heights = sorted(w.bbox[3] - w.bbox[1] for w in words)
            if raw_box[3] - raw_box[1] > 1.5 * (1 + 2 * PAD_Y) * heights[len(heights) // 2]:
                stats["statement_multirow_excluded"] += 1
                continue
            for cid, char in CID_MAP.items():
                if cid in gt_text:
                    stats["statement_cid_mapped"] += 1
                    gt_text = gt_text.replace(cid, char)
            box = _clip(raw_box, size)
            for cond in CONDITIONS:
                img = clean if cond == "clean" else _bgr(RUN_DIR / by_cond[cond]["image_path"])
                crop, page_box = _crop(img, box, hmat if cond == "photo" else None)
                lid = f"{sid}:{i}:{cond}"
                lines.append(
                    _record(
                        lid,
                        "statement",
                        sid,
                        by_cond[cond]["bank"],
                        cond,
                        gt_text,
                        crop,
                        by_cond[cond]["image_path"],
                        page_box,
                        size,
                        work,
                    )
                )

    # ThaiOCRBench Fine-grained: the question's region; single-line ground truth only.
    tob = sorted({r["sample_id"] for r in rows if r["subtask"] == "Fine-grained text recognition"})
    for sid in tob:
        by_cond = {r["condition"]: r for r in rows if r["sample_id"] == sid}
        gt_text = json.loads((RUN_DIR / by_cond["clean"]["gt_path"]).read_text())["text"]
        region = parse_region(by_cond["clean"]["question"])
        if region is None:
            stats["tob_fg_no_region"] += 1
            continue
        if "\n" in gt_text.strip():
            stats["tob_fg_multiline_excluded"] += 1
            continue
        clean = _bgr(RUN_DIR / by_cond["clean"]["image_path"])
        size = (clean.shape[1], clean.shape[0])
        box = region_to_pixels(region, size)
        _, hmat = degrade(clean, sid, Condition.photo)
        for cond in CONDITIONS:
            img = clean if cond == "clean" else _bgr(RUN_DIR / by_cond[cond]["image_path"])
            crop, page_box = _crop(img, box, hmat if cond == "photo" else None)
            lid = f"{sid}:0:{cond}"
            lines.append(
                _record(
                    lid,
                    "tob_fine_grained",
                    sid,
                    None,
                    cond,
                    gt_text,
                    crop,
                    by_cond[cond]["image_path"],
                    page_box,
                    size,
                    work,
                )
            )
        stats["tob_fg_lines"] += 1

    stats["tob_text_recognition_excluded"] = len(
        {r["sample_id"] for r in rows if r["subtask"] == "Text recognition"}
    )
    with open(work / "lines.jsonl", "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    stats["crops"] = len(lines)
    stats["statement_segments"] = sum(
        1 for x in lines if x["source"] == "statement" and x["condition"] == "clean"
    )
    stats["statement_numeric_segments"] = sum(
        1
        for x in lines
        if x["source"] == "statement" and x["condition"] == "clean" and x["numeric"]
    )
    (work / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


_DOTS_TEXT_RE = re.compile(r'"text":\s*"((?:[^"\\]|\\.)*)')


def _unescape(fragment: str) -> str:
    try:
        return json.loads(f'"{fragment}"')
    except ValueError:  # cut off inside an escape
        return fragment


def dots_grounding_text(raw: str) -> str:
    """dots.ocr answers a grounding prompt with layout JSON, `[{"bbox", "category", "text"}]`;
    the reading is its text fields joined. A reply cut off mid-JSON keeps the texts it has."""
    try:
        items = json.loads(raw)
    except ValueError:
        texts = [_unescape(m) for m in _DOTS_TEXT_RE.findall(raw)]
        return " ".join(texts) if texts else raw
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return raw
    return " ".join(str(i.get("text", "")) for i in items if isinstance(i, dict))


# --- read ----------------------------------------------------------------------------------


def _load_lines(work: Path) -> list[dict]:
    with open(work / "lines.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as fh:
        return {r["id"] for r in map(json.loads, fh) if r.get("error") is None}


async def _read(args) -> None:
    work = Path(args.work)
    spec = READERS[args.reader]
    cfg = load_model_config(Path(f"configs/models/{spec['model']}.yaml"), spec["model"])
    # Line reads go straight to the vLLM server; the PaddleX pipeline is not needed.
    model = build_model(cfg.model_copy(update={"pipeline": None}), base_url=args.base_url)
    version = await model.server_version()
    sampling = Sampling(**{**cfg.sampling.model_dump(), "max_tokens": LINE_MAX_TOKENS})
    prompt = spec["prompt"] or cfg.plans["ocr_line"].prompt
    out_path = work / f"read-{args.reader}.jsonl"
    done = _done(out_path)
    todo = [x for x in _load_lines(work) if x["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    # Same-page requests together, so vLLM's encoder and prefix caches reuse the page image.
    todo.sort(key=lambda x: (x["page"], x["id"]))
    print(f"{args.reader}: {len(todo)} to read ({len(done)} done), vLLM {version}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    fh = open(out_path, "a", encoding="utf-8")
    started = time.perf_counter()
    n = 0

    async def one(line: dict) -> None:
        nonlocal n
        async with sem:
            if args.reader == "dots_grounding":
                page = Image.open(RUN_DIR / line["page"]).convert("RGB")
                sent = model.preprocess(page)
                sx, sy = sent.size[0] / page.size[0], sent.size[1] / page.size[1]
                x0, y0, x1, y1 = line["box"]
                box = [round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)]
                text_prompt = f"{prompt}[{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
                orig = page.size
            else:
                crop = Image.open(work / line["crop"]).convert("RGB")
                work_img = resize_by_need(crop)  # the harness's crop guard (min edge 28)
                sent = model.preprocess(work_img)
                text_prompt = prompt
                orig = work_img.size
            result = await model.client.chat(sent, text_prompt, sampling)
            if result.error is not None:
                text, err = "", result.error
            else:
                ctx = ParseContext(task="ocr_line", orig_size=orig, sent_size=sent.size)
                raw = result.text
                if args.reader == "dots_grounding":
                    raw = dots_grounding_text(raw)
                text, err = parse(spec["parser"], raw, ctx).text, None
            rec = {
                "id": line["id"],
                "reader": args.reader,
                "text": text,
                "raw": result.text,
                "completion_tokens": result.completion_tokens,
                "latency_ms": round(result.latency_ms, 1),
                "error": err,
                "vllm": version,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 500 == 0:
                rate = n / (time.perf_counter() - started)
                print(f"  {n}/{len(todo)} ({rate:.1f} lines/s)", flush=True)

    await asyncio.gather(*(one(x) for x in todo))
    fh.close()
    elapsed = time.perf_counter() - started
    print(f"{args.reader}: read {n} lines in {elapsed:.0f} s", flush=True)


# --- report --------------------------------------------------------------------------------


def _fold(s: str) -> str:
    """NIKHAHIT + SARA AA -> SARA AM: some text layers spell the vowel decomposed, which
    renders identically (27 of the crops); `canonical_text` does not fold it."""
    return s.replace("\u0e4d\u0e32", "\u0e33")


def _nospace(s: str) -> str:
    return re.sub(r"\s+", "", plain_text(s))


def _cer_nospace(pred: str, gt: str) -> float:
    p, g = _nospace(pred), _nospace(gt)
    return min(1.0, Levenshtein.distance(p, g) / max(1, len(g)))


def _summary(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {"n": 0}
    pairs = [(_fold(p), _fold(g)) for p, g in pairs]
    cers = [cer(p, g) for p, g in pairs]
    ns = [_cer_nospace(p, g) for p, g in pairs]
    chars = sum(len(plain_text(g)) for _, g in pairs)
    edits = sum(Levenshtein.distance(plain_text(p), plain_text(g)) for p, g in pairs)
    return {
        "n": len(pairs),
        "cer_mean": round(statistics.fmean(cers), 4),
        "cer_micro": round(edits / max(1, chars), 4),
        "cer_nospace_mean": round(statistics.fmean(ns), 4),
        "exact": round(sum(plain_text(p) == plain_text(g) for p, g in pairs) / len(pairs), 4),
        "exact_nospace": round(sum(_nospace(p) == _nospace(g) for p, g in pairs) / len(pairs), 4),
    }


READER_SETTINGS = {
    "paddle_crop": "PaddleOCR-VL-1.6 vLLM, prompt 'OCR:' (the pipeline's text-block read), "
    "crop as is, T=0",
    "typhoon_crop": "typhoon-ocr1.5-2b vLLM, its one official prompt, typhoon_markdown parser, T=0",
    "dots_grounding": "dots.ocr vLLM, prompt_grounding_ocr on the whole page (smart_resize "
    "4 MP) with the line box; text fields of the JSON reply, T=0",
    "vlm_common": f"vLLM 0.22.1 bf16, max_tokens {LINE_MAX_TOKENS}, crops smaller than 28 px "
    "upscaled (resize_by_need)",
    "muocr_<prep>_nr<k>": "muocr-base-26m-stage2-finetuned-20240820-v1, transformers 4.43.2, "
    "fp32, beam 4, max_new_tokens 120, length_penalty 2.0; prep squash = resize to 64x384 "
    "(its ViTImageProcessor), pad = height 64 keeping aspect, pad right with the background; "
    "k = no_repeat_ngram_size",
}


def _aspect_bucket(line: dict) -> str:
    w, h = line["crop_size"]
    a = w / max(1, h)
    return "lt2" if a < 2 else "2to4" if a < 4 else "4to6" if a <= MAX_ASPECT + 0.05 else "gt6"


def cmd_report(args) -> None:
    work = Path(args.work)
    lines = {x["id"]: x for x in _load_lines(work)}
    reader_files = sorted(work.glob("read-*.jsonl"))
    readings: dict[str, dict[str, dict]] = {}
    for path in reader_files:
        name = path.stem.removeprefix("read-")
        with open(path, encoding="utf-8") as fh:
            by_id: dict[str, dict] = {}
            for rec in map(json.loads, fh):
                if rec.get("error") is None or rec["id"] not in by_id:
                    by_id[rec["id"]] = rec
        readings[name] = by_id

    # Only lines every reader has read: the gate compares readers on the same lines.
    common = set(lines)
    for by_id in readings.values():
        common &= set(by_id)
    report: dict = {
        "task": "docparse 1.1 (Spike A)",
        "run_id": RUN_DIR.name,
        "date": time.strftime("%Y-%m-%d"),
        "build": json.loads((work / "build_stats.json").read_text()),
        "segmentation": {
            "statement": "text-layer words -> group_lines -> split at gaps > "
            f"{GAP_FACTOR} line heights -> split at word boundaries to aspect <= {MAX_ASPECT}",
            "padding": {"y": PAD_Y, "x": PAD_X, "unit": "line height"},
            "photo": "quad mapped through the photo homography and rectified",
            "tob_fine_grained": "question region, single-line ground truth only",
        },
        "metric": "CER = min(1, Levenshtein / len(gt)) over ocr_bench plain_text; "
        "cer_nospace drops all whitespace first; *_mean is per line, cer_micro pools edits; "
        "NIKHAHIT+SARA AA folded to SARA AM on both sides",
        "lines_scored": len(common),
        "lines_missing": {k: len(set(lines) - set(v)) for k, v in readings.items()},
        "readers": {},
    }
    report["reader_settings"] = READER_SETTINGS
    tput = work / "muocr_throughput.json"
    if tput.exists():
        report["muocr_throughput"] = json.loads(tput.read_text())

    for name, by_id in sorted(readings.items()):
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        errors = 0
        for lid in sorted(common):
            line, rec = lines[lid], by_id[lid]
            errors += rec.get("error") is not None
            pair = (rec.get("text") or "", line["gt"])
            groups[f"all/{line['condition']}"].append(pair)
            groups[f"{line['source']}/{line['condition']}"].append(pair)
            if line["source"] == "statement":
                kind = "numeric" if line["numeric"] else "text"
                groups[f"statement_{kind}/{line['condition']}"].append(pair)
                groups[f"statement_aspect_{_aspect_bucket(line)}/{line['condition']}"].append(pair)
                groups[f"statement_bank_{line['bank']}/{line['condition']}"].append(pair)
        entry = {key: _summary(pairs) for key, pairs in sorted(groups.items())}
        entry["errors"] = errors
        entry["hit_max_tokens"] = sum(
            1 for lid in common if by_id[lid].get("completion_tokens", 0) >= LINE_MAX_TOKENS
        )
        report["readers"][name] = entry

    # Where muocr's two no_repeat_ngram_size settings read a line differently, which one wins.
    for prep in ("squash", "pad"):
        a, b = readings.get(f"muocr_{prep}_nr0"), readings.get(f"muocr_{prep}_nr3")
        if not (a and b):
            continue
        diff = [lid for lid in common if a[lid]["text"] != b[lid]["text"]]

        def _c(rec, lid):
            return cer(_fold(rec[lid]["text"]), _fold(lines[lid]["gt"]))

        report[f"muocr_{prep}_nr0_vs_nr3"] = {
            "lines_differ": len(diff),
            "nr0_better": sum(_c(a, lid) < _c(b, lid) for lid in diff),
            "nr3_better": sum(_c(a, lid) > _c(b, lid) for lid in diff),
            "numeric_lines_differ": sum(lines[lid]["numeric"] for lid in diff),
        }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {REPORT}")
    header = f"{'reader':28s}" + "".join(f"{c:>10s}" for c in CONDITIONS)
    for group in ("statement", "statement_numeric", "tob_fine_grained"):
        print(f"\n{group} (cer_mean)\n{header}")
        for name, entry in report["readers"].items():
            vals = [entry.get(f"{group}/{c}", {}).get("cer_mean") for c in CONDITIONS]
            print(
                f"{name:28s}"
                + "".join(f"{v:>10.4f}" if v is not None else f"{'-':>10s}" for v in vals)
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default=str(WORK))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    rd = sub.add_parser("read")
    rd.add_argument("--reader", choices=sorted(READERS), required=True)
    rd.add_argument("--base-url", required=True)
    rd.add_argument("--concurrency", type=int, default=64)
    rd.add_argument("--limit", type=int, default=0)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "read":
        asyncio.run(_read(args))
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
