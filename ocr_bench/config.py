"""Configuration models and loader for `run.yaml` + `models/*.yaml` + `datasets/*.yaml`."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ocr_bench.schemas import Condition, Task


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    version: str


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    endpoint_env: str
    served_model_name: str
    default_parser: str
    parsers: dict[Task, str] = Field(default_factory=dict)
    prompts: dict[Task, PromptSpec]


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Task
    cap: int | None = None


class ThaiOCRBenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["thaiocrbench"]
    hf_repo: str
    split: str = "test"
    seed: int = 42
    tasks: dict[str, TaskSpec]
    domain_field: str | None = "domain"
    priority_domains: list[str] = Field(default_factory=lambda: ["Government", "Finance"])


class RowLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str


class CorpusTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_digital_pages: int = 40
    min_scanned_photo_pages: int = 40
    min_banks: int = 4
    min_multipage_files: int = 10


class BankStmtConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["bankstmt"]
    input_dir: str
    dpi: int = 200
    banks: list[str]
    bank_map: dict[str, str]
    layouts: dict[str, RowLayout] = Field(default_factory=dict)
    targets: CorpusTargets = Field(default_factory=CorpusTargets)


DatasetConfig = Annotated[
    ThaiOCRBenchConfig | BankStmtConfig,
    Field(discriminator="kind"),
]
_DATASET_CONFIG_ADAPTER: TypeAdapter[DatasetConfig] = TypeAdapter(DatasetConfig)


class LatencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: list[int] = Field(default_factory=lambda: [1, 4, 8, 16, 32])
    mix_size: int = 200
    warmup_s: float = 30.0
    multipage_files: int = 10


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[str]
    datasets: list[str]
    conditions: list[Condition] = Field(
        default_factory=lambda: [Condition.clean, Condition.scan_low, Condition.photo]
    )
    concurrency: int = 8
    seed: int = 42
    bootstrap_resamples: int = 1000
    gpu_hourly_rate: dict[str, float] = Field(default_factory=dict)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)


class ResolvedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    models: dict[str, ModelConfig]
    datasets: dict[str, DatasetConfig]

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        )


class ConfigError(ValueError):
    pass


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(run_yaml: Path) -> ResolvedConfig:
    """Load and validate `run_yaml` plus its referenced model and dataset configs."""
    root = run_yaml.parent

    try:
        run_data = _load_yaml(run_yaml)
        run_config = RunConfig.model_validate(run_data)
    except ValidationError as exc:
        raise ConfigError(f"{run_yaml}: {exc}") from exc

    models: dict[str, ModelConfig] = {}
    for m in run_config.models:
        path = root / "models" / f"{m}.yaml"
        if not path.exists():
            raise ConfigError(f"model '{m}' has no config at {path}")
        data = _load_yaml(path)
        data["name"] = m
        try:
            models[m] = ModelConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"{path}: {exc}") from exc

    datasets: dict[str, DatasetConfig] = {}
    for d in run_config.datasets:
        path = root / "datasets" / f"{d}.yaml"
        if not path.exists():
            raise ConfigError(f"dataset '{d}' has no config at {path}")
        data = _load_yaml(path)
        try:
            datasets[d] = _DATASET_CONFIG_ADAPTER.validate_python(data)
        except ValidationError as exc:
            raise ConfigError(f"{path}: {exc}") from exc

    return ResolvedConfig(run=run_config, models=models, datasets=datasets)
