"""Build an `OcrModel` from a `ModelConfig` (adapter class by model name)."""

import os
from typing import Any

from ocr_bench.config import ModelConfig
from ocr_bench.models.base import ChatClient, PlannedModel
from ocr_bench.models.dotsocr import DotsOCRModel
from ocr_bench.models.pipeline_client import PipelineClient
from ocr_bench.models.teleocr import TeleOCRModel
from ocr_bench.models.typhoon import TyphoonModel
from ocr_bench.models.vllm_client import EndpointUnavailable, VllmClient

# Models without an entry (ovisocr2, paddleocr_vl) run as plain `PlannedModel`s.
ADAPTERS: dict[str, type[PlannedModel]] = {
    "teleocr": TeleOCRModel,
    "dotsocr": DotsOCRModel,
    "typhoon_ocr15": TyphoonModel,
}


def build_model(
    cfg: ModelConfig,
    *,
    client: ChatClient | None = None,
    openai_client: Any | None = None,
    base_url: str | None = None,
    pipeline: PipelineClient | None = None,
) -> PlannedModel:
    """Adapter for `cfg`. Without an explicit `client`, a `VllmClient` is built on
    `base_url` or `$<endpoint_env>`; likewise a `PipelineClient` on
    `$<pipeline.endpoint_env>` when the config has a pipeline. An unset variable raises
    `EndpointUnavailable`."""
    if client is None:
        url = base_url or os.environ.get(cfg.endpoint_env)
        if not url:
            raise EndpointUnavailable(cfg.name, f"${cfg.endpoint_env}", "variable is not set")
        client = VllmClient(
            url,
            cfg.served_model_name,
            system_prompt=cfg.system_prompt,
            message_prefix=cfg.message_prefix,
            client=openai_client,
            model_label=cfg.name,
        )
    if pipeline is None and cfg.pipeline is not None:
        url = os.environ.get(cfg.pipeline.endpoint_env)
        if not url:
            raise EndpointUnavailable(
                cfg.name, f"${cfg.pipeline.endpoint_env}", "variable is not set"
            )
        pipeline = PipelineClient(url, cfg.pipeline, model_label=cfg.name)
    return ADAPTERS.get(cfg.name, PlannedModel)(cfg, client, pipeline)
