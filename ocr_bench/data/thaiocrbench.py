"""ThaiOCRBench subset selection and ground-truth extraction (task 2.1).

`select` and `to_sample` are pure over plain metadata dicts / row mappings, so
they are tested without ever touching the network or the Hugging Face hub.
`load_rows` is the only function that talks to `datasets`, and it is never
called from tests; `prepare_thaiocrbench` accepts `rows` directly, or reads
them through `THAIOCRBENCH_ROWS_PROVIDER` when it is set (the tests'
monkeypatch seam).
"""

import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from math import floor
from typing import Any

import cv2
import numpy as np

from ocr_bench.config import TaskSpec, ThaiOCRBenchConfig
from ocr_bench.data.gt_utils import (
    critical_fields_for,
    flatten_fields,
    parse_answer_json,
    table_answer_to_html,
)
from ocr_bench.data.manifest import PreparedSample
from ocr_bench.schemas import GroundTruth, HtmlGT, JsonGT, Source, Task, TextGT

_TEXT_TASKS = {Task.ocr_fullpage, Task.ocr_line, Task.handwriting, Task.docparse}

THAIOCRBENCH_ROWS_PROVIDER: Callable[[ThaiOCRBenchConfig], Any] | None = None


class TaskNameError(ValueError):
    pass


def load_rows(cfg: ThaiOCRBenchConfig) -> Sequence[Mapping]:
    """Load `cfg.hf_repo`'s `cfg.split`, filtered to the configured task names.

    Never called in tests: `prepare_thaiocrbench` takes `rows` directly.
    """
    import datasets

    kept = set(cfg.tasks.keys())
    ds = datasets.load_dataset(cfg.hf_repo, split=cfg.split)
    return ds.filter(lambda t: t in kept, input_columns=["Task"])


def _stratum_key(row: Mapping, domain_field: str | None) -> str:
    if domain_field is None:
        return ""
    value = row.get(domain_field)
    return value if value else ""


def _stratified_sample(
    idxs_sorted: list[int],
    rows_meta: Sequence[Mapping],
    seed: int,
    task_name: str,
    domain_field: str | None,
    cap: int,
) -> list[int]:
    n = len(idxs_sorted)
    strata: dict[str, list[int]] = defaultdict(list)
    for idx in idxs_sorted:
        strata[_stratum_key(rows_meta[idx], domain_field)].append(idx)

    quotas: dict[str, int] = {}
    fracs: dict[str, float] = {}
    for cat, idxs in strata.items():
        exact = cap * len(idxs) / n
        base = floor(exact)
        quotas[cat] = base
        fracs[cat] = exact - base

    remainder = cap - sum(quotas.values())
    order = sorted(strata.keys(), key=lambda c: (-fracs[c], c))
    for cat in order[:remainder]:
        quotas[cat] += 1

    selected: list[int] = []
    for cat, idxs in strata.items():
        quota_c = quotas[cat]
        if quota_c <= 0:
            continue
        id_to_idx = {rows_meta[i]["Id"]: i for i in idxs}
        sorted_ids = sorted(id_to_idx.keys())
        chosen_ids = random.Random(f"{seed}|{task_name}|{cat}").sample(sorted_ids, quota_c)
        selected.extend(id_to_idx[cid] for cid in chosen_ids)
    return selected


def select(rows_meta: Sequence[Mapping], cfg: ThaiOCRBenchConfig) -> tuple[list[int], dict]:
    """Pick the indices (into `rows_meta`) to keep, applying name checks and caps."""
    present = {r["Task"] for r in rows_meta}
    missing = [n for n in cfg.tasks if n not in present]
    if missing:
        raise TaskNameError(
            f"ThaiOCRBench is missing expected task name(s) {missing}; present: {sorted(present)}"
        )

    by_task: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(rows_meta):
        if r["Task"] in cfg.tasks:
            by_task[r["Task"]].append(idx)

    domain_field = cfg.domain_field
    kept_indices = [idx for idxs in by_task.values() for idx in idxs]
    if domain_field is not None and kept_indices:
        domain_slice_available = all(
            rows_meta[idx].get(domain_field) not in (None, "") for idx in kept_indices
        )
    else:
        domain_slice_available = False

    warnings: list[str] = []
    if not domain_slice_available:
        warnings.append("ThaiOCRBench has no domain field; Government/Finance slice unavailable")

    effective_domain_field = domain_field if domain_slice_available else None

    selected: list[int] = []
    per_task_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = defaultdict(int)

    for task_name, idxs in by_task.items():
        spec = cfg.tasks[task_name]
        idxs_sorted = sorted(idxs, key=lambda i: rows_meta[i]["Id"])
        n = len(idxs_sorted)
        cap = spec.cap
        if cap is None or n <= cap:
            task_selected = idxs_sorted
        else:
            task_selected = _stratified_sample(
                idxs_sorted, rows_meta, cfg.seed, task_name, effective_domain_field, cap
            )
        selected.extend(task_selected)
        per_task_counts[task_name] = len(task_selected)
        if domain_slice_available:
            for idx in task_selected:
                domain_counts[_stratum_key(rows_meta[idx], effective_domain_field)] += 1

    info = {
        "thaiocrbench": per_task_counts,
        "thaiocrbench_domains": dict(domain_counts),
        "domain_slice_available": domain_slice_available,
        "warnings": warnings,
    }
    return selected, info


def to_sample(
    row: Mapping, spec: TaskSpec, task_name: str, domain_field: str | None
) -> PreparedSample:
    """Build one `PreparedSample` from a raw ThaiOCRBench row."""
    sample_id = f"TOB-{spec.code}-{row['Id']}"
    image = cv2.cvtColor(np.array(row["image"].convert("RGB")), cv2.COLOR_RGB2BGR)
    question = row["question"]
    answer = row["answer"]
    domain = row.get(domain_field) if domain_field is not None else None

    critical_fields: list[str] = []
    gt: GroundTruth
    if spec.task in _TEXT_TASKS:
        gt = TextGT(gt_kind="text", text=answer)
    elif spec.task == Task.table:
        gt = HtmlGT(gt_kind="html", html=table_answer_to_html(answer), raw=answer)
    elif spec.task in (Task.kie, Task.kie_map):
        try:
            parsed = parse_answer_json(answer)
        except ValueError as exc:
            raise ValueError(f"{sample_id}: {exc}") from exc
        fields = flatten_fields(parsed)
        critical_fields = critical_fields_for(fields)
        gt = JsonGT(gt_kind="json", fields=fields, raw=answer)
    elif spec.task == Task.classify:
        gt = JsonGT(gt_kind="json", label=answer.strip(), raw=answer)
    else:
        raise ValueError(f"unsupported ThaiOCRBench task: {spec.task}")

    return PreparedSample(
        sample_id=sample_id,
        source=Source.thaiocrbench,
        task=spec.task,
        subtask=task_name,
        domain=domain,
        bank=None,
        doc_type=None,
        page_no=1,
        n_pages=1,
        question=question,
        image=image,
        gt=gt,
        critical_fields=critical_fields,
    )


def prepare_thaiocrbench(
    cfg: ThaiOCRBenchConfig, rows: Sequence[Mapping] | None = None
) -> tuple[list[PreparedSample], dict]:
    """Select and materialize ThaiOCRBench samples for `cfg`."""
    if rows is None:
        if THAIOCRBENCH_ROWS_PROVIDER is not None:
            rows = THAIOCRBENCH_ROWS_PROVIDER(cfg)
        else:
            rows = load_rows(cfg)

    if hasattr(rows, "select_columns"):
        rows_meta: Sequence[Mapping] = list(rows.select_columns(["Id", "Task", "category"]))
    else:
        rows_meta = list(rows)

    selected_indices, info = select(rows_meta, cfg)
    domain_field = cfg.domain_field if info["domain_slice_available"] else None

    samples = []
    for idx in selected_indices:
        task_name = rows_meta[idx]["Task"]
        spec = cfg.tasks[task_name]
        samples.append(to_sample(rows[idx], spec, task_name, domain_field))

    return samples, info
