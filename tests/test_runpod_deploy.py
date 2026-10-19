"""`ocrbench runpod deploy` (design D4/D9) against a recorded fake Runpod REST v2 API."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.runpod import cli as runpod_cli
from ocr_bench.runpod import deploy as dp
from ocr_bench.runpod.api import RunpodAPI, RunpodAPIError
from ocr_bench.runpod.config import config_hash, load_runpod_config
from ocr_bench.runpod.releases import load_releases
from ocr_bench.runpod.render import image_repo, template_name
from ocr_bench.runpod.state import load_state
from tests.runpod_fake_api import BLACKWELL_MIG, FakeRunpod
from tests.runpod_helpers import REV, pinned_configs

runner = CliRunner()
DIGEST = "sha256:" + "c" * 64
DIGEST2 = "sha256:" + "e" * 64
KEY = "test-management-key"
ALLOWED = ["EU-RO-1", "EU-SE-1"]
CRED = "cred-test-id"


@pytest.fixture
def configs(tmp_path) -> Path:
    cdir = pinned_configs(tmp_path / "c")
    (cdir / "data_centers.yaml").write_text(yaml.safe_dump({"data_centers": ALLOWED}))
    _edit(cdir, data_centers=["EU-RO-1"])
    # The registry credential (task 5.2) sits in both state paths the tests use.
    for state_path in (tmp_path / ".runpod" / "state.yaml", tmp_path / "state.yaml"):
        _write_state(state_path, {})
    return cdir


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"registry_credential_id": CRED, **state}))


def _no_endpoints(path: Path) -> bool:
    return not load_state(path).get("endpoints")


def _edit(cdir: Path, model: str = "dotsocr", **endpoint) -> None:
    path = cdir / f"{model}.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["endpoint"].update(endpoint)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


def _cfg(cdir: Path, model: str = "dotsocr"):
    return load_runpod_config(cdir / f"{model}.yaml")


def _api(fake: FakeRunpod) -> RunpodAPI:
    return RunpodAPI(KEY, transport=fake.transport())


def _deploy(cdir, fake, tmp_path, digest=DIGEST, model="dotsocr", **kw):
    return dp.deploy(
        _cfg(cdir, model),
        digest,
        _api(fake),
        allowed_data_centers=dp.load_allowed_data_centers(cdir / "data_centers.yaml"),
        state_path=tmp_path / ".runpod" / "state.yaml",
        releases_path=tmp_path / "releases.yaml",
        git_sha="a" * 40,
        deployer="tester",
        today=date(2026, 10, 15),
        **kw,
    )


# --- the API client -------------------------------------------------------------------------


def test_api_sends_bearer_key_to_v2_paths():
    fake = FakeRunpod(list_wrapper="data")
    api = _api(fake)
    assert api.list_templates() == []
    assert api.list_endpoints() == []
    assert [c[1] for c in fake.calls] == ["/v2/templates", "/v2/serverless"]
    assert fake.auth == [f"Bearer {KEY}"] * 2
    assert str(api.base_url) == "https://api.runpod.io/v2"


def test_api_error_names_status_but_not_key():
    api = RunpodAPI(KEY, transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no")))
    with pytest.raises(RunpodAPIError) as err:
        api.list_endpoints()
    assert "401" in str(err.value)
    assert KEY not in str(err.value)


def test_api_from_env_requires_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RunpodAPIError, match="RUNPOD_API_KEY"):
        RunpodAPI.from_env()


# --- deploy -----------------------------------------------------------------------------------


def test_first_deploy_creates_template_and_endpoint(configs, tmp_path):
    fake = FakeRunpod()
    cfg = _cfg(configs)
    result = _deploy(configs, fake, tmp_path)

    assert result.action == "created"
    methods = [(m, p) for m, p, _ in fake.writes]
    assert methods == [("POST", "/v2/templates"), ("POST", "/v2/serverless")]
    tmpl_body = fake.writes[0][2]
    assert tmpl_body["name"] == f"ocrbench-dotsocr-{config_hash(cfg)[:8]}"
    assert tmpl_body["image"] == f"{image_repo('dotsocr')}@{DIGEST}"
    ep_body = fake.writes[1][2]
    assert ep_body["name"] == "ocrbench-dotsocr"
    assert ep_body["templateId"] == fake.templates[0]["id"]
    assert ep_body["type"] == "LOAD_BALANCER"
    assert ep_body["gpu"] == {
        "pools": ["AMPERE_24"],
        "excludedTypes": [BLACKWELL_MIG],  # in the pool but not in gpu_types
        "count": 1,
        "minCudaVersion": "12.8",
    }
    assert ep_body["dataCenterIds"] == ["EU-RO-1"]
    assert ep_body["workers"] == {"min": 0, "max": 3, "idleTimeout": 120}
    assert ep_body["scaling"] == {"type": "REQUEST_COUNT", "requestCount": 4}
    assert ep_body["flashboot"] == "FLASHBOOT"
    assert fake.endpoints[0]["image"] == tmpl_body["image"]  # applied from the template

    entry = load_state(tmp_path / ".runpod" / "state.yaml")["endpoints"]["ocrbench-dotsocr"]
    assert entry["id"] == fake.endpoints[0]["id"]
    assert entry["template_id"] == fake.templates[0]["id"]
    assert entry["model"] == "dotsocr"
    assert entry["client_data"] is False

    (rec,) = load_releases(tmp_path / "releases.yaml")
    assert rec == {
        "model": "dotsocr",
        "action": "deploy",
        "date": "2026-10-15",
        "git_sha": "a" * 40,
        "image_digest": DIGEST,
        "config_hash": config_hash(cfg),
        "weights_revision": REV,
        "vllm_version": "0.11.0",
        "endpoint_kind": "load_balancing",
        "deployer": "tester",
        "parity": "pending",
    }
    # no ids in the committed log
    text = (tmp_path / "releases.yaml").read_text()
    assert fake.endpoints[0]["id"] not in text
    assert fake.templates[0]["id"] not in text


def test_repeat_deploy_is_a_noop(configs, tmp_path):
    fake = FakeRunpod()
    _deploy(configs, fake, tmp_path)
    fake.calls.clear()
    result = _deploy(configs, fake, tmp_path)
    assert result.action == "up_to_date"
    assert fake.writes == []
    assert len(load_releases(tmp_path / "releases.yaml")) == 1


def test_scaling_only_change_patches_endpoint_without_new_template(configs, tmp_path):
    fake = FakeRunpod()
    _deploy(configs, fake, tmp_path)
    # The config hash changes, but the template body (image, disk, ports) does not, so the
    # existing template is reused and only the endpoint settings are patched.
    _edit(configs, workers_min=1)
    fake.calls.clear()
    result = _deploy(configs, fake, tmp_path)
    assert result.action == "patched"
    assert [(m, p) for m, p, _ in fake.writes] == [
        ("PATCH", f"/v2/serverless/{fake.endpoints[0]['id']}")
    ]
    assert fake.writes[0][2] == {"workers": {"min": 1, "max": 3, "idleTimeout": 120}}
    assert len(fake.templates) == 1
    assert len(load_releases(tmp_path / "releases.yaml")) == 1


def test_new_digest_creates_template_and_repoints_endpoint(configs, tmp_path):
    fake = FakeRunpod()
    _deploy(configs, fake, tmp_path)
    fake.calls.clear()
    result = _deploy(configs, fake, tmp_path, digest=DIGEST2)
    assert result.action == "updated"
    assert [(m, p) for m, p, _ in fake.writes] == [
        ("POST", "/v2/templates"),
        ("PATCH", f"/v2/serverless/{fake.endpoints[0]['id']}"),
    ]
    new_tmpl = fake.templates[1]
    assert new_tmpl["image"].endswith(DIGEST2)
    assert new_tmpl["name"] != fake.templates[0]["name"]  # template names stay unique
    assert fake.writes[1][2] == {"templateId": new_tmpl["id"]}
    recs = load_releases(tmp_path / "releases.yaml")
    assert [r["image_digest"] for r in recs] == [DIGEST, DIGEST2]


def test_new_release_suspends_promotion(configs, tmp_path):
    fake = FakeRunpod()
    _deploy(configs, fake, tmp_path)
    state_path = tmp_path / ".runpod" / "state.yaml"
    state = load_state(state_path)
    state["endpoints"]["ocrbench-dotsocr"].update(
        client_data=True, approved_by="reviewer", approved_on="2026-10-01"
    )
    state_path.write_text(yaml.safe_dump(state))
    _deploy(configs, fake, tmp_path, digest=DIGEST2)
    entry = load_state(state_path)["endpoints"]["ocrbench-dotsocr"]
    assert entry["client_data"] is False
    assert entry["approved_by"] == "reviewer"


def test_existing_endpoint_is_found_by_name(configs, tmp_path):
    cfg = _cfg(configs)
    fake = FakeRunpod(
        templates=[{"id": "t0", "name": template_name(cfg), "image": "x@" + DIGEST2}],
        endpoints=[{"id": "e0", "name": "ocrbench-dotsocr", "type": "LOAD_BALANCER"}],
        list_wrapper="data",
    )
    result = _deploy(configs, fake, tmp_path)
    assert result.endpoint_id == "e0"
    assert not any(p == "/v2/serverless" and m == "POST" for m, p, _ in fake.writes)
    entry = load_state(tmp_path / ".runpod" / "state.yaml")["endpoints"]["ocrbench-dotsocr"]
    assert entry["id"] == "e0"


def test_kind_mismatch_is_refused_and_changes_nothing(configs, tmp_path):
    fake = FakeRunpod(endpoints=[{"id": "e0", "name": "ocrbench-dotsocr", "type": "QUEUE"}])
    with pytest.raises(dp.DeployRefused) as err:
        _deploy(configs, fake, tmp_path)
    msg = str(err.value)
    assert "ocrbench-dotsocr" in msg and "queue" in msg and "load_balancing" in msg
    assert fake.writes == []
    assert not (tmp_path / "releases.yaml").exists()


def test_unknown_remote_kind_is_refused(configs, tmp_path):
    fake = FakeRunpod(endpoints=[{"id": "e0", "name": "ocrbench-dotsocr"}])
    with pytest.raises(dp.DeployRefused, match="kind"):
        _deploy(configs, fake, tmp_path)
    assert fake.writes == []


# --- refusals: each one deploys nothing -------------------------------------------------------


def _refused(configs, tmp_path, match, digest=DIGEST, fake=None):
    fake = fake or FakeRunpod()
    with pytest.raises(dp.DeployRefused, match=match):
        _deploy(configs, fake, tmp_path, digest=digest)
    assert fake.writes == []
    assert not (tmp_path / "releases.yaml").exists()
    assert _no_endpoints(tmp_path / ".runpod" / "state.yaml")


def test_refuses_without_a_registry_credential(configs, tmp_path):
    (tmp_path / ".runpod" / "state.yaml").write_text(yaml.safe_dump({}))
    _refused(configs, tmp_path, "registry_credential_id")


def test_cli_dry_run_refuses_without_a_registry_credential(configs, tmp_path):
    (tmp_path / "state.yaml").unlink()
    result = _cli(configs, tmp_path, "--digest", DIGEST, "--dry-run")
    assert result.exit_code == 2
    assert "registry_credential_id" in result.output


def test_created_template_uses_the_registry_credential(configs, tmp_path):
    fake = FakeRunpod()
    _deploy(configs, fake, tmp_path)
    assert fake.templates[0]["registry"] == CRED


def test_refuses_a_gpu_type_without_a_serverless_pool(configs, tmp_path):
    _edit(configs, gpu_types=["NVIDIA L4", "NVIDIA GeForce RTX 3080"])
    _refused(configs, tmp_path, "RTX 3080")


def test_refuses_a_gpu_type_missing_from_the_catalog(configs, tmp_path):
    _edit(configs, gpu_types=["NVIDIA L4", "NVIDIA Imaginary 9000"])
    _refused(configs, tmp_path, "Imaginary 9000")


def test_repeat_deploy_ignores_fields_the_api_adds(configs, tmp_path):
    fake = FakeRunpod()
    _deploy(configs, fake, tmp_path)
    ep = fake.endpoints[0]
    ep["gpu"] = {**ep["gpu"], "allowedCudaVersions": []}  # the live API returns these too
    ep["mounts"] = {"network": [], "global": []}
    ep["timeout"] = 300000
    fake.calls.clear()
    assert _deploy(configs, fake, tmp_path).action == "up_to_date"
    assert fake.writes == []


def test_refuses_empty_data_centers(configs, tmp_path):
    _edit(configs, data_centers=[])
    _refused(configs, tmp_path, "data_centers is empty")


def test_refuses_data_center_outside_allow_list(configs, tmp_path):
    _edit(configs, data_centers=["EU-RO-1", "US-TX-3"])
    _refused(configs, tmp_path, "US-TX-3")


def test_refuses_when_allow_list_is_empty(configs, tmp_path):
    (configs / "data_centers.yaml").write_text("data_centers: []\n")
    _refused(configs, tmp_path, "EU-RO-1")


@pytest.mark.parametrize("digest", ["latest", "ghcr.io/x/runpod-dotsocr:abc", "sha256:abc"])
def test_refuses_a_tag(configs, tmp_path, digest):
    _refused(configs, tmp_path, "digest", digest=digest)


def test_accepts_a_repo_qualified_digest(configs, tmp_path):
    fake = FakeRunpod()
    _deploy(configs, fake, tmp_path, digest=f"{image_repo('dotsocr')}@{DIGEST}")
    assert fake.templates[0]["image"] == f"{image_repo('dotsocr')}@{DIGEST}"


def test_refuses_digest_of_another_models_image(configs, tmp_path):
    _refused(configs, tmp_path, "runpod-typhoon", digest=f"{image_repo('typhoon_ocr15')}@{DIGEST}")


def test_refuses_argv_that_reenables_logging(configs, tmp_path):
    path = configs / "dotsocr.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["vllm_args"]["enable-log-outputs"] = True
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    _refused(configs, tmp_path, "--enable-log-outputs")


def test_refuses_argv_without_d9_flags(configs, tmp_path, monkeypatch):
    real = dp.render_vllm_argv
    monkeypatch.setattr(
        dp, "render_vllm_argv", lambda cfg: [a for a in real(cfg) if a != "--max-log-len"]
    )
    _refused(configs, tmp_path, "--max-log-len")


def test_logging_flag_checker():
    from ocr_bench.runpod.config import logging_flags

    good = ["--seed", "0", *logging_flags("0.22.1")]
    assert dp.logging_flag_problems(good, "0.22.1") == []
    assert dp.logging_flag_problems(good, "0.11.0") == ["missing --disable-log-requests"]
    bad = [*good, "--max-log-len", "100"]  # the last occurrence wins in vllm's argparse
    assert dp.logging_flag_problems(bad, "0.22.1") == ["--max-log-len must be 0, got 100"]


def test_refuses_network_volume_on_promotable_config(configs, tmp_path):
    _edit(configs, network_volume="vol-shared")
    _refused(configs, tmp_path, "vol-shared")


def test_refuses_remote_network_volume_on_promotable_config(configs, tmp_path):
    fake = FakeRunpod(
        endpoints=[
            {
                "id": "e0",
                "name": "ocrbench-dotsocr",
                "type": "LOAD_BALANCER",
                "mounts": {"network": [{"volumeId": "vol-console"}], "global": []},
            }
        ]
    )
    _refused(configs, tmp_path, "vol-console", fake=fake)


def test_network_volume_counts_in_config_hash_only_when_set(configs):
    before = config_hash(_cfg(configs))
    _edit(configs, network_volume="vol-shared")
    assert config_hash(_cfg(configs)) != before
    _edit(configs, network_volume=None)
    assert config_hash(_cfg(configs)) == before


def test_network_volume_allowed_on_queue_config(configs, tmp_path):
    _edit(configs, kind="queue", network_volume="vol-shared")
    fake = FakeRunpod()
    result = _deploy(configs, fake, tmp_path)
    assert result.action == "created"
    ep_body = fake.writes[1][2]
    assert ep_body["type"] == "QUEUE"
    assert ep_body["mounts"] == {"network": [{"volumeId": "vol-shared"}]}


def test_network_volume_refused_on_promoted_queue_endpoint(configs, tmp_path):
    _edit(configs, kind="queue", network_volume="vol-shared")
    _write_state(
        tmp_path / ".runpod" / "state.yaml",
        {"endpoints": {"ocrbench-dotsocr": {"id": "e0", "client_data": True}}},
    )
    fake = FakeRunpod()
    with pytest.raises(dp.DeployRefused, match="vol-shared"):
        _deploy(configs, fake, tmp_path)
    assert fake.writes == []


# --- the CLI ----------------------------------------------------------------------------------


def _cli(configs, tmp_path, *args):
    return runner.invoke(
        app,
        [
            "runpod",
            "deploy",
            "dotsocr",
            "--configs-dir",
            str(configs),
            "--state",
            str(tmp_path / "state.yaml"),
            "--releases",
            str(tmp_path / "releases.yaml"),
            *args,
        ],
    )


def test_cli_dry_run_prints_bodies_without_network(configs, tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(runpod_cli, "make_api", lambda: pytest.fail("dry run made an API"))
    result = _cli(configs, tmp_path, "--digest", DIGEST, "--dry-run")
    assert result.exit_code == 0, result.output
    bodies = json.loads(result.output)
    assert bodies["template"]["image"].endswith(DIGEST)
    assert bodies["template"]["registry"] == CRED
    assert bodies["endpoint"]["type"] == "LOAD_BALANCER"
    assert bodies["endpoint"]["gpu"]["pools"] == ["<from /v2/catalog/gpus at deploy>"]
    assert bodies["endpoint"]["templateId"] == "<template-id>"
    assert _no_endpoints(tmp_path / "state.yaml")


def test_cli_dry_run_still_applies_refusals(configs, tmp_path):
    _edit(configs, data_centers=[])
    result = _cli(configs, tmp_path, "--digest", DIGEST, "--dry-run")
    assert result.exit_code == 2
    assert "data_centers" in result.output


def test_cli_deploy_then_noop(configs, tmp_path, monkeypatch):
    fake = FakeRunpod()
    monkeypatch.setattr(runpod_cli, "make_api", lambda: _api(fake))
    first = _cli(configs, tmp_path, "--digest", DIGEST)
    assert first.exit_code == 0, first.output
    assert "created" in first.output
    assert fake.endpoints[0]["id"] not in first.output  # ids stay in the state file
    second = _cli(configs, tmp_path, "--digest", DIGEST)
    assert second.exit_code == 0, second.output
    assert "up to date" in second.output


def test_cli_refusal_exits_2(configs, tmp_path, monkeypatch):
    fake = FakeRunpod(endpoints=[{"id": "e0", "name": "ocrbench-dotsocr", "type": "QUEUE"}])
    monkeypatch.setattr(runpod_cli, "make_api", lambda: _api(fake))
    result = _cli(configs, tmp_path, "--digest", DIGEST)
    assert result.exit_code == 2
    assert "queue" in result.output
