"""Build plan + workflow checks for runpod/ci/runpod-images.yml (task 2.4).

The workflow is installed as .github/workflows/runpod-images.yml by the runner.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from ocr_bench.runpod.build_plan import affected_models, build_matrix

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "runpod" / "ci" / "runpod-images.yml"
CONFIGS = ROOT / "configs" / "runpod"


def test_runpod_code_change_affects_every_model():
    models = affected_models(["runpod/Dockerfile"], CONFIGS)
    assert {"dotsocr", "teleocr", "paddleocr_vl"} <= set(models)


def test_every_planned_model_has_a_loadable_config(tmp_path):
    shutil.copytree(CONFIGS.parent / "models", tmp_path / "models")
    configs = shutil.copytree(CONFIGS, tmp_path / "runpod")
    (configs / "releases.yaml").write_text("- {model: dotsocr, action: deploy}\n")
    models = affected_models(["runpod/Dockerfile"], configs)
    assert "releases" not in models
    assert len(build_matrix(models, configs)) == len(models)


def test_single_config_change_affects_only_that_model():
    assert affected_models(["configs/runpod/dotsocr.yaml"], CONFIGS) == ["dotsocr"]


def test_non_model_configs_are_ignored():
    assert affected_models(["configs/runpod/data_centers.yaml"], CONFIGS) == []
    assert affected_models(["configs/runpod/releases.yaml"], CONFIGS) == []  # deploy's log
    assert affected_models(["README.md"], CONFIGS) == []


def test_matrix_entry_carries_build_args(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.yaml").write_text("{}")
    cfg = yaml.safe_load((CONFIGS / "dotsocr.yaml").read_text())
    cfg["model"] = "m"
    cfg["weights"]["revision"] = "a" * 40
    d = tmp_path / "runpod"
    d.mkdir()
    (d / "m.yaml").write_text(yaml.safe_dump(cfg))
    (entry,) = build_matrix(["m"], d)
    assert entry == {
        "model": "m",
        "variant": "plain",
        "base": cfg["image"]["base"],
        "repo": cfg["weights"]["repo"],
        "revision": "a" * 40,
    }
    json.dumps(entry)


def _wf():
    wf = yaml.safe_load(WORKFLOW.read_text())
    wf["on"] = wf.pop(True, wf.get("on"))  # YAML reads bare `on` as True
    return wf


def test_workflow_triggers():
    on = _wf()["on"]
    assert on["push"]["branches"] == ["main"]
    assert set(on["push"]["paths"]) == {"runpod/**", "configs/runpod/**"}
    assert on["workflow_dispatch"]["inputs"]["model"]["required"] is True


def test_workflow_builds_pushes_and_summarises():
    text = WORKFLOW.read_text()
    assert "docker/build-push-action" in text
    assert "code.loolootech.com/loolootech/runpod-" in text
    assert "secrets.LOOLOO_REGISTRY_USER" in text
    assert "secrets.LOOLOO_REGISTRY_TOKEN" in text
    assert "github.sha" in text
    assert "HF_TOKEN=${{ secrets.HF_TOKEN }}" in text
    assert "GITHUB_STEP_SUMMARY" in text
    assert "steps.build.outputs.digest" in text
    assert _wf()["permissions"] == {"contents": "read"}  # the registry is not GHCR


@pytest.mark.skipif(shutil.which("actionlint") is None, reason="actionlint not installed")
def test_workflow_passes_actionlint():
    subprocess.run(["actionlint", str(WORKFLOW)], check=True)
