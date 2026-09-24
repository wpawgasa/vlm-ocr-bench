"""Tests for docparse.artifacts: hashing, freshness and atomic writes."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from docparse.artifacts import (
    ArtifactStore,
    MissingInputError,
    config_hash,
    doc_id_for_bytes,
    doc_id_for_file,
)
from docparse.schemas import (
    ArtifactMeta,
    ClassifyArtifact,
    DocStatus,
    LayoutArtifact,
    ModelIdentity,
    ReadingsArtifact,
    ReconcileArtifact,
    SegmentsArtifact,
    VerifyArtifact,
)

DOC_BYTES = b"fake statement"

STAGES = ("classify", "layout", "segments", "readings", "reconcile", "verify")


def _compute(stage: str, cfg: dict):
    """Build a compute() callable for `stage` given a fake config dict."""

    def compute(meta: ArtifactMeta) -> BaseModel:
        if stage == "classify":
            return ClassifyArtifact(
                meta=meta, class_id="bank_statement", confidence=0.9, pages_used=[1]
            )
        if stage == "layout":
            return LayoutArtifact(
                meta=meta, model="paddleocr_vl", prompt_version=cfg["v"], pages=[]
            )
        if stage == "segments":
            return SegmentsArtifact(meta=meta, max_aspect=cfg["max_aspect"], segments=[])
        if stage == "readings":
            return ReadingsArtifact(meta=meta, readings=[])
        if stage == "reconcile":
            return ReconcileArtifact(meta=meta, segments=[])
        if stage == "verify":
            return VerifyArtifact(meta=meta, vlm_calls=cfg["budget"])
        raise AssertionError(stage)

    return compute


def run_all(store: ArtifactStore, configs: dict[str, dict], counters: dict[str, int] | None = None):
    """Run all six stages in order via run_stage; return the set of stages that ran."""
    ran: set[str] = set()
    for stage in STAGES:
        cfg = configs[stage]

        def compute(meta: ArtifactMeta, _stage=stage, _cfg=cfg) -> BaseModel:
            if counters is not None:
                counters[_stage] = counters.get(_stage, 0) + 1
            return _compute(_stage, _cfg)(meta)

        _, did_run = store.run_stage(stage, cfg, compute)
        if did_run:
            ran.add(stage)
    return ran


def default_configs() -> dict[str, dict]:
    return {
        "classify": {},
        "layout": {"v": "v1"},
        "segments": {"max_aspect": 5.0},
        "readings": {},
        "reconcile": {},
        "verify": {"budget": 3},
    }


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path, doc_id_for_bytes(DOC_BYTES))


# 1. doc_id hashing


def test_doc_id_for_bytes_is_16_hex_chars():
    doc_id = doc_id_for_bytes(DOC_BYTES)
    assert len(doc_id) == 16
    assert doc_id == doc_id.lower()
    int(doc_id, 16)  # raises if not hex


def test_doc_id_for_bytes_deterministic_and_distinguishing():
    assert doc_id_for_bytes(DOC_BYTES) == doc_id_for_bytes(DOC_BYTES)
    assert doc_id_for_bytes(DOC_BYTES) != doc_id_for_bytes(b"other bytes")


def test_doc_id_for_file_matches_bytes(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_bytes(DOC_BYTES)
    assert doc_id_for_file(path) == doc_id_for_bytes(DOC_BYTES)


# 2. config_hash


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_value():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_config_hash_model_matches_its_dump():
    identity = ModelIdentity(
        role="layout", model="paddleocr_vl", served_model_name="x", prompt_version="v1"
    )
    assert config_hash(identity) == config_hash(identity.model_dump(mode="json"))


# 3. first run does all stages, second run does none


def test_run_all_first_time_runs_everything(store: ArtifactStore):
    ran = run_all(store, default_configs())
    assert ran == set(STAGES)


def test_run_all_second_time_runs_nothing(store: ArtifactStore):
    configs = default_configs()
    run_all(store, configs)
    counters: dict[str, int] = {}
    ran = run_all(store, configs, counters)
    assert ran == set()
    assert counters == {}


# 4. changing only verify config re-runs only verify


def test_verify_config_change_reruns_only_verify(store: ArtifactStore):
    configs = default_configs()
    run_all(store, configs)
    configs["verify"] = {"budget": 9}
    ran = run_all(store, configs)
    assert ran == {"verify"}


# 5. changing layout config re-runs layout and everything downstream, not classify


def test_layout_config_change_reruns_downstream_not_classify(store: ArtifactStore):
    configs = default_configs()
    run_all(store, configs)
    configs["layout"] = {"v": "v2"}
    ran = run_all(store, configs)
    assert ran == {"layout", "segments", "readings", "reconcile", "verify"}


# 6. changing segments config re-runs segments downstream only


def test_segments_config_change_reruns_segments_downstream_only(store: ArtifactStore):
    configs = default_configs()
    run_all(store, configs)
    configs["segments"] = {"max_aspect": 8.0}
    ran = run_all(store, configs)
    assert ran == {"segments", "readings", "reconcile", "verify"}


# 7. identical artifact rewrite keeps downstream fresh


def test_identical_rewrite_keeps_downstream_fresh(store: ArtifactStore):
    configs = default_configs()
    run_all(store, configs)

    before = store.input_hashes("segments")
    layout_artifact = store.read("layout")
    assert layout_artifact is not None
    store.write("layout", layout_artifact)  # rewrite the identical artifact object
    after = store.input_hashes("segments")

    assert before == after
    cfg_hash = config_hash(configs["segments"])
    assert store.is_fresh("segments", cfg_hash) is True


# 8. run_stage before an input exists raises MissingInputError


def test_run_stage_missing_input_raises(store: ArtifactStore):
    configs = default_configs()
    with pytest.raises(MissingInputError) as exc_info:
        store.run_stage("segments", configs["segments"], _compute("segments", configs["segments"]))
    message = str(exc_info.value)
    assert "segments" in message
    assert "layout" in message


# 9. corrupt artifact file is treated as absent


def test_corrupt_artifact_triggers_rerun(store: ArtifactStore):
    configs = default_configs()
    run_all(store, configs)

    store.dir.mkdir(parents=True, exist_ok=True)
    store.path("verify").write_text("{not json", encoding="utf-8")
    assert store.read("verify") is None

    ran = run_all(store, configs)
    assert ran == {"verify"}


# 10. write() rejects mismatches


def test_write_rejects_wrong_model(store: ArtifactStore):
    meta = ArtifactMeta(stage="classify", doc_id=store.doc_id, config_hash="x")
    wrong = LayoutArtifact(meta=meta, model="m", prompt_version="v1", pages=[])
    with pytest.raises(ValueError):
        store.write("classify", wrong)


def test_write_rejects_stage_mismatch(store: ArtifactStore):
    meta = ArtifactMeta(stage="layout", doc_id=store.doc_id, config_hash="x")
    artifact = LayoutArtifact(meta=meta, model="m", prompt_version="v1", pages=[])
    with pytest.raises(ValueError):
        store.write("classify", artifact)  # meta.stage is "layout", stage arg is "classify"


def test_write_rejects_doc_id_mismatch(store: ArtifactStore):
    meta = ArtifactMeta(stage="classify", doc_id="ffffffffffffffff", config_hash="x")
    artifact = ClassifyArtifact(
        meta=meta, class_id="bank_statement", confidence=0.5, pages_used=[1]
    )
    with pytest.raises(ValueError):
        store.write("classify", artifact)


# 11. run_stage validates the returned meta


def test_run_stage_rejects_config_hash_change(store: ArtifactStore):
    configs = default_configs()

    def bad_compute(meta: ArtifactMeta) -> BaseModel:
        tampered = meta.model_copy(update={"config_hash": "tampered"})
        return ClassifyArtifact(
            meta=tampered, class_id="bank_statement", confidence=0.5, pages_used=[1]
        )

    with pytest.raises(ValueError):
        store.run_stage("classify", configs["classify"], bad_compute)


def test_run_stage_accepts_models_addition(store: ArtifactStore):
    configs = default_configs()
    identity = ModelIdentity(
        role="classifier", model="qwen3vl", served_model_name="Qwen3-VL", prompt_version="v1"
    )

    def compute(meta: ArtifactMeta) -> BaseModel:
        meta_with_models = meta.model_copy(update={"models": [identity]})
        return ClassifyArtifact(
            meta=meta_with_models, class_id="bank_statement", confidence=0.5, pages_used=[1]
        )

    artifact, ran = store.run_stage("classify", configs["classify"], compute)
    assert ran is True
    assert artifact.meta.models == [identity]


# 12. atomic writes leave no tmp files; write_text/write_model name validation


def test_atomic_writes_leave_no_tmp_files(store: ArtifactStore):
    configs = default_configs()
    run_all(store, configs)
    store.write_text("parsed.html", "<html></html>")
    store.write_model("status.json", DocStatus(doc_id=store.doc_id, status="complete"))

    tmp_files = list(store.dir.glob("*.tmp"))
    assert tmp_files == []


def test_write_text_rejects_path_traversal(store: ArtifactStore):
    with pytest.raises(ValueError):
        store.write_text("../x.html", "x")


def test_write_text_rejects_nested_path(store: ArtifactStore):
    with pytest.raises(ValueError):
        store.write_text("a/b.html", "x")


# 13. invalid doc_id


def test_invalid_doc_id_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        ArtifactStore(tmp_path, "XYZ")
