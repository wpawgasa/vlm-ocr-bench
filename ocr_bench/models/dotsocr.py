"""dots.ocr adapter. Everything model-specific is config (prompts, `<|img|>` message prefix,
smart_resize pre-resize); layout bboxes are mapped from the sent (resized) image back to
original pixels by the `dots_layout_json` parser via `ParseContext.scale`."""

from ocr_bench.models.base import PlannedModel


class DotsOCRModel(PlannedModel):
    pass
