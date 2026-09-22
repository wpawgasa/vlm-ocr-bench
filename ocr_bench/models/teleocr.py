"""TeleOCR adapter: the official two-stage pipeline is config-driven (`two_stage` plans in
configs/models/teleocr.yaml, executed by `PlannedModel`); this module adds TeleOCR's block
post-processing from its reference client."""

from ocr_bench.models.base import PlannedModel
from ocr_bench.models.parsers import fix_equation, is_otsl, otsl_to_html


class TeleOCRModel(PlannedModel):
    def postprocess_block(self, block_type: str, text: str) -> str:
        # Table (and chart->table "char") blocks answer in OTSL; anything else is kept
        # verbatim rather than converted into an empty table.
        if block_type in ("table", "char") and is_otsl(text):
            return otsl_to_html(text)
        if block_type == "equation":
            return fix_equation(text)
        return text
