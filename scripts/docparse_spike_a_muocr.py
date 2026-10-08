"""docparse Spike A (task 1.1): read the line set with the muocr TrOCR checkpoint on the GPU.

Standalone (torch, transformers, sentencepiece, pillow) so it runs inside the vLLM image, e.g.

    docker run --rm --gpus '"device=0"' --entrypoint python3 -v "$PWD:/w" -w /w \
        -v <deps>:/deps -e PYTHONPATH=/deps \
        vllm/vllm-openai:v0.22.1 scripts/docparse_spike_a_muocr.py

where <deps> holds the checkpoint's transformers (`pip install --target <deps>
transformers==4.43.2 sentencepiece`).

Variants (design D7): the input either `squash` (the checkpoint's own ViTImageProcessor:
resize straight to 64x384) or `pad` (keep the aspect ratio at height 64, pad right with the
background colour; a crop wider than 6:1 is scaled to fit 384 wide and padded below), crossed
with `no_repeat_ngram_size` 3 (the checkpoint default) and 0. Every variant decodes with beam
4, max_new_tokens 120 and the checkpoint's other generation settings.

Writes <work>/read-muocr_<prep>_nr<k>.jsonl and <work>/muocr_throughput.json (segments per
second with nr0 for each preparation at several batch sizes, decode and preprocessing).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import MarianTokenizer, VisionEncoderDecoderModel

H, W = 64, 384


def bg_colour(img: np.ndarray) -> tuple[int, int, int]:
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return tuple(int(v) for v in np.median(edge, axis=0))


def prep_pad(img: Image.Image) -> Image.Image:
    arr = np.asarray(img)
    w, h = img.size
    scale = min(H / h, W / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (W, H), bg_colour(arr))
    canvas.paste(img.resize((nw, nh), Image.BICUBIC), (0, 0 if nh == H else (H - nh) // 2))
    return canvas


def prep_squash(img: Image.Image) -> Image.Image:
    return img.resize((W, H), Image.BILINEAR)  # ViTImageProcessor resample=2 (bilinear)


PREP = {"squash": prep_squash, "pad": prep_pad}


def to_tensor(images: list[Image.Image], device) -> torch.Tensor:
    arr = np.stack([np.asarray(i, dtype=np.float32) for i in images]) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)


def decode(model, tok, pixels, nr: int) -> list[str]:
    with torch.inference_mode():
        out = model.generate(
            pixels,
            num_beams=4,
            max_new_tokens=120,
            no_repeat_ngram_size=nr,
            early_stopping=True,
            length_penalty=2.0,
        )
    return tok.batch_decode(out, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="runs/docparse/spikes/line_a")
    ap.add_argument("--checkpoint", default="models/muocr-base-26m-stage2-finetuned-20240820-v1")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument(
        "--only",
        choices=["read", "throughput"],
        default=None,
        help="run one phase; throughput needs the GPU to itself",
    )
    args = ap.parse_args()
    work = Path(args.work)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    tok = MarianTokenizer.from_pretrained(args.checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint, torch_dtype=dtype)
    # transformers 5.x builds TrOCR's sinusoidal table under the meta-device init context and
    # it is not in the checkpoint; rebuild it if so (a no-op on the checkpoint's own 4.43).
    pos = model.decoder.model.decoder.embed_positions
    if getattr(pos, "weights", None) is not None and pos.weights.is_meta:
        pos.weights = pos.get_embedding(pos.weights.shape[0], pos.embedding_dim, pos.padding_idx)
    model.to(device).eval()
    model.generation_config.use_cache = True

    with open(work / "lines.jsonl", encoding="utf-8") as fh:
        lines = [json.loads(x) for x in fh]
    crops = [Image.open(work / x["crop"]).convert("RGB") for x in lines]
    print(
        f"{len(lines)} lines, torch {torch.__version__}, {torch.cuda.get_device_name()}", flush=True
    )

    for prep in ("pad", "squash") if args.only != "throughput" else ():
        images = [PREP[prep](c) for c in crops]
        for nr in (0, 3):
            name = f"muocr_{prep}_nr{nr}"
            t0 = time.perf_counter()
            texts: list[str] = []
            for i in range(0, len(images), args.batch):
                pixels = to_tensor(images[i : i + args.batch], device).to(dtype)
                texts.extend(decode(model, tok, pixels, nr))
            elapsed = time.perf_counter() - t0
            with open(work / f"read-{name}.jsonl", "w", encoding="utf-8") as fh:
                for line, text in zip(lines, texts, strict=True):
                    rec = {"id": line["id"], "reader": name, "text": text, "error": None}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"{name}: {len(texts)} lines in {elapsed:.1f} s ({len(texts) / elapsed:.1f}/s)",
                flush=True,
            )

    if args.only == "read":
        return
    # Throughput (nr0, beam 4) on the statement segments of one condition, for each input
    # preparation and batch size; each batch size starts with a warm-up batch.
    stmt = [
        c
        for c, x in zip(crops, lines, strict=True)
        if x["source"] == "statement" and x["condition"] == "clean"
    ]
    by_prep: dict[str, list[dict]] = {}
    for prep, fn in PREP.items():
        results = by_prep.setdefault(prep, [])
        for bs in (32, 64, 128, 256):
            decode(model, tok, to_tensor([fn(c) for c in stmt[:bs]], device).to(dtype), 0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prep_s = 0.0
            for i in range(0, len(stmt), bs):
                p0 = time.perf_counter()
                pixels = to_tensor([fn(c) for c in stmt[i : i + bs]], device).to(dtype)
                prep_s += time.perf_counter() - p0
                decode(model, tok, pixels, 0)
            torch.cuda.synchronize()
            total = time.perf_counter() - t0
            results.append(
                {
                    "batch": bs,
                    "segments": len(stmt),
                    "seconds": round(total, 2),
                    "segments_per_s": round(len(stmt) / total, 1),
                    "preprocess_s": round(prep_s, 2),
                    "s_per_300_segments": round(300 * total / len(stmt), 2),
                }
            )
            print(prep, results[-1], flush=True)
    meta = {
        "setting": "no_repeat_ngram_size 0, beam 4, max_new_tokens 120",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "gpu_shared": False,
        "by_prep": by_prep,
    }
    (work / "muocr_throughput.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
