from pathlib import Path

import pytest
import yaml

from ocr_bench.runpod.parity_set import MINIMUMS, load_parity_set
from ocr_bench.schemas import Condition, GtKind, ManifestRow, Source, Task

COMMITTED = Path(__file__).resolve().parents[1] / "configs" / "runpod" / "parity_samples.v1.yaml"


def _row(
    sample_id: str,
    task: Task,
    source: Source = Source.thaiocrbench,
    condition: Condition = Condition.clean,
) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        source=source,
        task=task,
        condition=condition,
        image_path=f"img/{sample_id}.png",
        gt_kind=GtKind.text,
    )


def _samples() -> dict[str, list[str]]:
    return {t.value: [f"TOB-{t.value}-{i:04d}" for i in range(n)] for t, n in MINIMUMS.items()}


def _rows(samples: dict[str, list[str]]) -> list[ManifestRow]:
    return [_row(i, Task(t)) for t, ids in samples.items() for i in ids]


def _write(tmp_path: Path, samples: dict[str, list[str]]) -> Path:
    path = tmp_path / "parity.yaml"
    path.write_text(
        yaml.safe_dump({"version": 1, "run_id": "2026-09-22-a", "samples": samples}),
        encoding="utf-8",
    )
    return path


def test_valid_set_loads(tmp_path):
    samples = _samples()
    ps = load_parity_set(_write(tmp_path, samples), _rows(samples))
    assert ps.version == 1
    assert ps.run_id == "2026-09-22-a"
    assert len(ps.sample_ids) == 60


def test_bankstmt_id_is_refused(tmp_path):
    samples = _samples()
    rows = _rows(samples)
    rows[0] = _row(rows[0].sample_id, Task.ocr_fullpage, source=Source.bankstmt)
    with pytest.raises(ValueError, match="bankstmt"):
        load_parity_set(_write(tmp_path, samples), rows)


def test_below_per_task_minimum_is_refused(tmp_path):
    samples = _samples()
    samples["kie"] = samples["kie"][:9]
    with pytest.raises(ValueError, match="kie"):
        load_parity_set(_write(tmp_path, samples), _rows(samples))


def test_unknown_id_is_refused(tmp_path):
    samples = _samples()
    rows = _rows(samples)[1:]
    with pytest.raises(ValueError, match="not in the manifest"):
        load_parity_set(_write(tmp_path, samples), rows)


def test_id_without_clean_row_is_refused(tmp_path):
    samples = _samples()
    rows = _rows(samples)
    rows[0] = _row(rows[0].sample_id, Task.ocr_fullpage, condition=Condition.photo)
    with pytest.raises(ValueError, match="clean"):
        load_parity_set(_write(tmp_path, samples), rows)


def test_task_mismatch_with_manifest_is_refused(tmp_path):
    samples = _samples()
    rows = _rows(samples)
    rows[0] = _row(rows[0].sample_id, Task.table)
    with pytest.raises(ValueError, match="task"):
        load_parity_set(_write(tmp_path, samples), rows)


def test_committed_file_contains_ids_only():
    raw = yaml.safe_load(COMMITTED.read_text(encoding="utf-8"))
    assert set(raw) == {"version", "run_id", "samples"}
    assert raw["run_id"] == "2026-09-22-a"
    counts = {t: len(ids) for t, ids in raw["samples"].items()}
    assert counts == {t.value: n for t, n in MINIMUMS.items()}
    ids = [i for v in raw["samples"].values() for i in v]
    assert len(ids) == len(set(ids)) == 60
    assert all(isinstance(i, str) and i.startswith("TOB-") for i in ids)
    assert not any(i.startswith("BS-") for i in ids)
