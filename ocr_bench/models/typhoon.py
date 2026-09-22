"""typhoon-ocr1.5-2b adapter. Its single official prompt and the 1800 px max-side resize are
config; `<figure>`/`<page_number>` handling lives in the `typhoon_markdown` parser."""

from ocr_bench.models.base import PlannedModel


class TyphoonModel(PlannedModel):
    pass
