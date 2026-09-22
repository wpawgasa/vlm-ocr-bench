"""Configuration models and loader for `run.yaml` + `models/*.yaml` + `datasets/*.yaml`."""

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from ocr_bench.schemas import Condition, Task


class Sampling(BaseModel):
    """Decoding parameters of one request type. `top_k`, `repetition_penalty` and
    `vllm_xargs` are vLLM extensions sent through `extra_body`."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float | None = None
    vllm_xargs: dict[str, Any] = Field(default_factory=dict)


class TwoStage(BaseModel):
    """TeleOCR's official decoupled pipeline: one layout call, then one call per block."""

    model_config = ConfigDict(extra="forbid")

    layout_prompt: str
    layout_size: tuple[int, int] = (1036, 1036)
    layout_sampling: Sampling
    block_prompts: dict[str, str]
    block_sampling: dict[str, Sampling]
    skip_types: list[str]
    min_edge: int = 28
    max_edge_ratio: float = 50
    max_pixels: int = 64_000_000
    block_concurrency: int = 8

    @model_validator(mode="after")
    def _check_defaults(self) -> "TwoStage":
        if "default" not in self.block_prompts:
            raise ValueError("two_stage.block_prompts must contain 'default'")
        if "default" not in self.block_sampling:
            raise ValueError("two_stage.block_sampling must contain 'default'")
        return self


PlanKind = Literal["single", "crop_single", "grounding", "two_stage", "question"]


class RequestPlan(BaseModel):
    """How one model handles one (task, subtask): request shape, prompt, parser, version."""

    model_config = ConfigDict(extra="forbid")

    kind: PlanKind
    version: str
    prompt: str | None = None
    parser: str
    sampling: Sampling | None = None
    two_stage: TwoStage | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> "RequestPlan":
        if (self.kind == "two_stage") != (self.two_stage is not None):
            raise ValueError("`two_stage` must be set if and only if kind is 'two_stage'")
        if self.kind in ("single", "crop_single", "grounding") and not self.prompt:
            raise ValueError(f"plan kind '{self.kind}' requires a prompt")
        return self


# Plans are keyed by task, or by "task@subtask" to override one subtask.
FINE_GRAINED_PLAN_KEY = "ocr_line@Fine-grained text recognition"
REQUIRED_PLAN_KEYS = [t.value for t in Task] + [FINE_GRAINED_PLAN_KEY]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    endpoint_env: str
    served_model_name: str
    system_prompt: str | None = None
    message_prefix: str = ""
    sampling: Sampling = Field(default_factory=Sampling)
    image_resize: Literal["none", "smart_resize", "max_side"] = "none"
    max_pixels: int | None = None
    max_side: int | None = None
    plans: dict[str, RequestPlan]

    @model_validator(mode="after")
    def _check_plans(self) -> "ModelConfig":
        task_values = {t.value for t in Task}
        for key in self.plans:
            if key.split("@", 1)[0] not in task_values:
                raise ValueError(f"plan key '{key}' does not name a task")
        missing = [k for k in REQUIRED_PLAN_KEYS if k not in self.plans]
        if missing:
            raise ValueError(f"model '{self.name}' has no plan for: {missing}")
        if self.image_resize == "smart_resize" and self.max_pixels is None:
            raise ValueError("image_resize 'smart_resize' requires max_pixels")
        if self.image_resize == "max_side" and self.max_side is None:
            raise ValueError("image_resize 'max_side' requires max_side")
        return self

    def plan_for(self, task: Task | str, subtask: str | None) -> RequestPlan:
        task_value = Task(task).value
        if subtask is not None:
            plan = self.plans.get(f"{task_value}@{subtask}")
            if plan is not None:
                return plan
        try:
            return self.plans[task_value]
        except KeyError as exc:  # pragma: no cover — the validator guarantees coverage
            raise ConfigError(f"model '{self.name}' has no plan for {task_value}") from exc


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Task
    code: str
    cap: int | None = None


class ThaiOCRBenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["thaiocrbench"]
    hf_repo: str
    split: str = "test"
    seed: int = 42
    tasks: dict[str, TaskSpec]
    domain_field: str | None = "category"
    priority_domains: list[str] = Field(default_factory=lambda: ["Government", "Finance"])


class BankOverrides(BaseModel):
    """Per-bank details the page's own words cannot reveal (task 5.2).

    Row parsing is bank-agnostic: these only override what a layout cannot show.
    `date_format` is a `datetime.strptime` format tried before the generic date parser,
    `signed_amount` says a single amount column carries its sign (TTB), and
    `amount_column_split` allows a combined "Withdrawal / Deposit" column to take the
    side from the x-position of the amount (KBank).
    """

    model_config = ConfigDict(extra="forbid")

    date_format: str | None = None
    signed_amount: bool = False
    amount_column_split: bool = True


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
    overrides: dict[str, BankOverrides] = Field(default_factory=dict)
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

    def to_prepare_yaml(self) -> str:
        """The `config.resolved.yaml` written by `prepare`: run and datasets, no models."""
        return yaml.safe_dump(
            self.model_dump(mode="json", include={"run", "datasets"}),
            sort_keys=False,
            allow_unicode=True,
        )


class ConfigError(ValueError):
    pass


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_model_config(path: Path, name: str) -> ModelConfig:
    """Load and validate one `models/<name>.yaml`; raises `ConfigError` naming the file."""
    if not path.exists():
        raise ConfigError(f"model '{name}' has no config at {path}")
    data = _load_yaml(path)
    # Top-level `x-*` keys only hold YAML anchors (as in docker compose files).
    data = {k: v for k, v in data.items() if not str(k).startswith("x-")}
    data["name"] = name
    try:
        return ModelConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


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
        models[m] = load_model_config(path, m)

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
