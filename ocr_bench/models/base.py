"""`OcrModel` protocol and `PlannedModel`, which executes a model's request plans.

A request plan (config `RequestPlan`) says how one model handles one (task, subtask):
`single`, `crop_single`, `grounding`, `question` (one request each) or `two_stage`
(TeleOCR: one layout request, then one request per block). Whatever the plan, a page
yields exactly one `PredictionRow` whose `latency_ms` is the wall clock from the first
request's start to the last request's end, with tokens summed over its requests.

`predict` never raises: any failure becomes a row with `raw.error` set.
"""

import asyncio
import re
from typing import Protocol, runtime_checkable

from PIL import Image

from ocr_bench.config import ModelConfig, RequestPlan, Sampling
from ocr_bench.models.imaging import (
    Region,
    crop_block,
    crop_region,
    downscale_if_huge,
    region_to_pixels,
    resize_by_need,
    resize_max_side,
    rotate_crop,
    smart_resize,
)
from ocr_bench.models.parsers import (
    PARSERS,
    ParseContext,
    parse,
    parse_teleocr_layout,
    teleocr_page,
)
from ocr_bench.models.vllm_client import CallResult
from ocr_bench.schemas import ManifestRow, NormalizedPage, PredictionRow, RawCall, RawPrediction

NO_REGION = "no region in question"
WHOLE_IMAGE: Region = (0, 0, 1000, 1000)
_REGION_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def parse_region(question: str) -> Region | None:
    """The last `[x1, y1, x2, y2]` in a fine-grained question (0-1000 units), or None."""
    matches = _REGION_RE.findall(question)
    if not matches:
        return None
    x1, y1, x2, y2 = (int(v) for v in matches[-1])
    return x1, y1, x2, y2


@runtime_checkable
class OcrModel(Protocol):
    name: str

    async def health(self) -> None: ...

    async def predict(self, row: ManifestRow, image: Image.Image) -> PredictionRow: ...


class ChatClient(Protocol):
    async def health(self) -> None: ...

    async def chat(
        self, image: Image.Image, prompt: str, sampling: Sampling, *, stream: bool = False
    ) -> CallResult: ...


def _raw_call(result: CallResult, kind: str, index: int | None = None, block_type=None):
    return RawCall(
        kind=kind,
        block_index=index,
        block_type=block_type,
        text=result.text,
        tokens=result.tokens,
        token_logprobs=result.token_logprobs,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        ttft_ms=result.ttft_ms,
        error=result.error,
    )


class PlannedModel:
    """Runs a `ModelConfig`'s request plans against a chat client.

    Subclasses (`teleocr.py`, `dotsocr.py`, `typhoon.py`) only override hooks; request
    shapes, prompts, pre-resize and sampling all come from the config.
    """

    def __init__(self, cfg: ModelConfig, client: ChatClient):
        unknown = sorted({p.parser for p in cfg.plans.values()} - set(PARSERS))
        if unknown:
            raise ValueError(f"model '{cfg.name}' names unknown parser(s): {unknown}")
        self.cfg = cfg
        self.client = client
        self.name = cfg.name

    # --- hooks ------------------------------------------------------------------------------

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Model-level pre-resize of an image before it is sent."""
        if self.cfg.image_resize == "smart_resize":
            w, h = image.size
            new_h, new_w = smart_resize(h, w, max_pixels=self.cfg.max_pixels or 11_289_600)
            if (new_w, new_h) != (w, h):
                image = image.resize((new_w, new_h), Image.BICUBIC)
        elif self.cfg.image_resize == "max_side":
            image = resize_max_side(image, self.cfg.max_side or max(image.size))
        return image

    def postprocess_block(self, block_type: str, text: str) -> str:
        """Post-process one two-stage block's text before it enters the normalized page."""
        return text

    # --- entry points -----------------------------------------------------------------------

    async def health(self) -> None:
        await self.client.health()

    async def predict(self, row: ManifestRow, image: Image.Image) -> PredictionRow:
        return await self.predict_with_plan(row, image, self.cfg.plan_for(row.task, row.subtask))

    async def predict_with_plan(
        self, row: ManifestRow, image: Image.Image, plan: RequestPlan
    ) -> PredictionRow:
        try:
            image = image.convert("RGB")
            if plan.kind == "two_stage":
                return await self._two_stage(row, image, plan)
            return await self._single(row, image, plan)
        except Exception as exc:  # never drop a page
            return self._row(
                row,
                plan,
                calls=[],
                raw_text="",
                normalized=NormalizedPage(),
                error=f"{type(exc).__name__}: {exc}",
            )

    # --- plan kinds -------------------------------------------------------------------------

    async def _single(self, row: ManifestRow, image: Image.Image, plan: RequestPlan):
        note: str | None = None
        region: Region = WHOLE_IMAGE
        if plan.kind in ("crop_single", "grounding"):
            parsed = parse_region(row.question)
            if parsed is None:
                note = NO_REGION
            else:
                region = parsed

        work = image
        if plan.kind == "crop_single":
            # Crops get the same minimum-edge / aspect guard as TeleOCR block crops.
            work = resize_by_need(crop_region(image, region))
        sent = self.preprocess(work)

        if plan.kind == "question":
            prompt = row.question
        elif plan.kind == "grounding":
            x1, y1, x2, y2 = region_to_pixels(region, sent.size)
            prompt = f"{plan.prompt}[{x1}, {y1}, {x2}, {y2}]"
        else:
            prompt = plan.prompt or ""

        result = await self.client.chat(sent, prompt, plan.sampling or self.cfg.sampling)
        calls = [(_raw_call(result, "single"), result)]
        if result.error is not None:
            return self._row(row, plan, calls, "", NormalizedPage(), result.error)

        ctx = ParseContext(task=row.task, orig_size=work.size, sent_size=sent.size)
        normalized = parse(plan.parser, result.text, ctx)
        if note is not None:
            normalized.parse_error = (
                note if normalized.parse_error is None else f"{note}; {normalized.parse_error}"
            )
        return self._row(row, plan, calls, result.text, normalized, None)

    async def _two_stage(self, row: ManifestRow, image: Image.Image, plan: RequestPlan):
        ts = plan.two_stage
        assert ts is not None  # guaranteed by RequestPlan validation
        original = downscale_if_huge(image, ts.max_pixels)
        layout_image = original.resize(tuple(ts.layout_size), Image.BICUBIC)
        layout = await self.client.chat(layout_image, ts.layout_prompt, ts.layout_sampling)
        calls: list[tuple[RawCall, CallResult]] = [(_raw_call(layout, "layout"), layout)]
        if layout.error is not None:
            error = f"layout request failed: {layout.error}"
            return self._row(row, plan, calls, "", NormalizedPage(), error)

        items, warnings = parse_teleocr_layout(layout.text)
        targets = [item for item in items if item.label not in ts.skip_types]
        semaphore = asyncio.Semaphore(ts.block_concurrency)

        async def run_block(item) -> CallResult:
            try:
                crop = crop_block(original, item.points)
                crop = rotate_crop(crop, item.angle)
                crop = resize_by_need(crop, ts.min_edge, ts.max_edge_ratio)
            except Exception as exc:
                return CallResult(error=f"crop failed: {type(exc).__name__}: {exc}")
            prompt = ts.block_prompts.get(item.label, ts.block_prompts["default"])
            sampling = ts.block_sampling.get(item.label, ts.block_sampling["default"])
            async with semaphore:
                return await self.client.chat(crop, prompt, sampling)

        results = await asyncio.gather(*(run_block(item) for item in targets))

        texts: dict[int, str] = {}
        raw_texts: list[str] = []
        failed = 0
        for item, result in zip(targets, results, strict=True):
            calls.append((_raw_call(result, "block", item.index, item.label), result))
            if result.error is not None:
                failed += 1
                continue
            texts[item.index] = self.postprocess_block(item.label, result.text)
            if result.text:
                raw_texts.append(result.text)

        normalized = teleocr_page(items, texts, ParseContext(orig_size=image.size), ts.skip_types)
        if not items and layout.text.strip():
            detail = f": {warnings[0]}" if warnings else ""
            normalized.parse_error = f"no valid layout blocks{detail}"
        error = f"{failed} of {len(targets)} blocks failed" if failed else None
        return self._row(row, plan, calls, "\n\n".join(raw_texts), normalized, error)

    # --- row assembly -----------------------------------------------------------------------

    def _row(
        self,
        row: ManifestRow,
        plan: RequestPlan,
        calls: list[tuple[RawCall, CallResult]],
        raw_text: str,
        normalized: NormalizedPage,
        error: str | None,
    ) -> PredictionRow:
        timed = [r for _, r in calls if r.ended > 0]
        latency_ms = (
            (max(r.ended for r in timed) - min(r.started for r in timed)) * 1000.0 if timed else 0.0
        )
        prompt_tokens = sum(r.prompt_tokens for _, r in calls)
        completion_tokens = sum(r.completion_tokens for _, r in calls)
        raw = RawPrediction(
            text=raw_text,
            tokens=[t for c, _ in calls for t in c.tokens],
            token_logprobs=[lp for c, _ in calls for lp in c.token_logprobs],
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error,
            calls=[c for c, _ in calls],
        )
        return PredictionRow(
            sample_id=row.sample_id,
            condition=row.condition,
            model=self.name,
            prompt_version=plan.version,
            raw=raw,
            normalized=normalized,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            n_requests=len(calls),
        )
