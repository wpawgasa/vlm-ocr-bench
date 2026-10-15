"""`ocrbench runpod rollback|status|deploy --profile|--promote` (design D4/D7/D9)."""

from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.runpod import cli as runpod_cli
from ocr_bench.runpod import deploy as dp
from ocr_bench.runpod import manage
from ocr_bench.runpod.releases import load_releases
from ocr_bench.runpod.review import ReviewError, load_review
from ocr_bench.runpod.state import load_state
from tests.runpod_fake_api import FakeRunpod
from tests.test_runpod_deploy import (  # noqa: F401
    DIGEST,
    DIGEST2,
    _api,
    _cfg,
    _deploy,
    _edit,
    configs,
)

TODAY = date(2026, 10, 15)
NAME = "ocrbench-dotsocr"


def _review(path: Path, decision="approved", on="2026-09-01", **extra) -> Path:
    front = {
        "date": on,
        "reviewer": "Reviewer",
        "documents": ["DPA"],
        "decision": decision,
        "conditions": [],
        "next_review": "2027-09-01",
        **extra,
    }
    path.write_text(f"---\n{yaml.safe_dump(front)}---\n\n# Review\n", encoding="utf-8")
    return path


@pytest.fixture
def deployed(configs, tmp_path):  # noqa: F811
    fake = FakeRunpod()
    _deploy(configs, fake, tmp_path)
    return fake


def _paths(tmp_path):
    return {
        "state_path": tmp_path / ".runpod" / "state.yaml",
        "releases_path": tmp_path / "releases.yaml",
    }


def _set_parity(tmp_path, value="pass"):
    path = tmp_path / "releases.yaml"
    recs = load_releases(path)
    recs[-1]["parity"] = value
    path.write_text(yaml.safe_dump(recs, sort_keys=False))


def _promote(configs, fake, tmp_path, review, retention=None):  # noqa: F811
    return manage.promote(
        _cfg(configs),
        _api(fake),
        **_paths(tmp_path),
        review_path=review,
        approver="boss",
        today=TODAY,
        retention=retention,
    )


def _rollback(configs, fake, tmp_path):  # noqa: F811
    return manage.rollback(
        _cfg(configs),
        _api(fake),
        **_paths(tmp_path),
        git_sha="b" * 40,
        deployer="tester",
        today=TODAY,
    )


# --- review.py ---------------------------------------------------------------------------------


def test_load_review_parses_front_matter(tmp_path):
    r = load_review(_review(tmp_path / "r.md"))
    assert r.decision == "approved" and r.reviewer == "Reviewer"
    assert r.reviewed_on == date(2026, 9, 1)
    assert r.problem(TODAY) is None


def test_review_problems(tmp_path):
    assert "approved" in load_review(_review(tmp_path / "a.md", decision="rejected")).problem(TODAY)
    old = load_review(_review(tmp_path / "b.md", on="2025-10-14"))
    assert "12 months" in old.problem(TODAY)
    assert load_review(_review(tmp_path / "c.md", on="2025-10-15")).problem(TODAY) is None


def test_load_review_rejects_missing_or_malformed(tmp_path):
    with pytest.raises(ReviewError, match="r.md"):
        load_review(tmp_path / "r.md")
    (tmp_path / "bad.md").write_text("no front matter")
    with pytest.raises(ReviewError):
        load_review(tmp_path / "bad.md")
    (tmp_path / "nodate.md").write_text("---\ndecision: approved\n---\n")
    with pytest.raises(ReviewError, match="date"):
        load_review(tmp_path / "nodate.md")


# --- rollback ----------------------------------------------------------------------------------


def test_rollback_repoints_to_previous_template(configs, tmp_path, deployed):  # noqa: F811
    first_template = deployed.templates[0]["id"]
    _deploy(configs, deployed, tmp_path, digest=DIGEST2)
    deployed.calls.clear()
    result = _rollback(configs, deployed, tmp_path)
    assert result.template_id == first_template
    assert [(m, p) for m, p, _ in deployed.writes] == [
        ("PATCH", f"/v2/serverless/{deployed.endpoints[0]['id']}")
    ]
    assert deployed.writes[0][2] == {"templateId": first_template}
    recs = load_releases(tmp_path / "releases.yaml")
    assert [r["action"] for r in recs] == ["deploy", "deploy", "rollback"]
    assert recs[-1]["image_digest"] == DIGEST
    assert recs[-1]["parity"] == "pending"
    entry = load_state(tmp_path / ".runpod" / "state.yaml")["endpoints"][NAME]
    assert entry["template_id"] == first_template and entry["client_data"] is False


def test_rollback_without_previous_release_is_refused(configs, tmp_path, deployed):  # noqa: F811
    with pytest.raises(manage.ManageRefused, match="previous"):
        _rollback(configs, deployed, tmp_path)
    assert len(deployed.writes) == 2  # only the original deploy's writes


def test_rollback_refused_when_previous_template_is_gone(configs, tmp_path, deployed):  # noqa: F811
    _deploy(configs, deployed, tmp_path, digest=DIGEST2)
    deployed.templates.pop(0)
    n = len(deployed.writes)
    with pytest.raises(manage.ManageRefused, match="template"):
        _rollback(configs, deployed, tmp_path)
    assert len(deployed.writes) == n
    assert len(load_releases(tmp_path / "releases.yaml")) == 2


# --- status ------------------------------------------------------------------------------------


def _status(configs, fake, tmp_path, review=None, today=TODAY):  # noqa: F811
    return manage.status(
        _cfg(configs), _api(fake), **_paths(tmp_path), review_path=review, today=today
    )


def test_status_reports_endpoint_release_and_workers(configs, tmp_path, deployed):  # noqa: F811
    st = _status(configs, deployed, tmp_path)
    assert st.endpoint_name == NAME
    assert "…" in st.masked_id
    assert deployed.endpoints[0]["id"] not in st.masked_id
    assert st.parity == "pending" and st.promoted is False
    assert st.workers == {"min": 0, "max": 3}
    assert st.release_digest == DIGEST


def test_status_of_undeployed_model(configs, tmp_path):  # noqa: F811
    st = _status(configs, FakeRunpod(), tmp_path)
    assert st.masked_id is None and st.promoted is False


def test_status_flags_expired_review(configs, tmp_path, deployed):  # noqa: F811
    _set_parity(tmp_path)
    review = _review(tmp_path / "r.md")
    _promote(configs, deployed, tmp_path, review)
    st = _status(configs, deployed, tmp_path, review, today=date(2027, 10, 1))
    assert st.promoted and st.approver == "boss" and st.approved_on == "2026-10-15"
    assert st.review_problem and "12 months" in st.review_problem


# --- promote -----------------------------------------------------------------------------------


def test_promote_success_sets_state_and_mirrors_in_log(configs, tmp_path, deployed):  # noqa: F811
    _set_parity(tmp_path)
    _promote(configs, deployed, tmp_path, _review(tmp_path / "r.md"))
    entry = load_state(tmp_path / ".runpod" / "state.yaml")["endpoints"][NAME]
    assert entry["client_data"] is True
    assert entry["approved_by"] == "boss" and entry["approved_on"] == "2026-10-15"
    rec = load_releases(tmp_path / "releases.yaml")[-1]
    assert rec["client_data"] is True and rec["approved_by"] == "boss"
    assert rec["approved_on"] == "2026-10-15" and rec["parity"] == "pass"
    assert deployed.endpoints[0]["id"] not in (tmp_path / "releases.yaml").read_text()


def _refused_promote(configs, fake, tmp_path, match, review=None, **kw):  # noqa: F811
    review = review or _review(tmp_path / "r.md")
    with pytest.raises(manage.ManageRefused, match=match):
        _promote(configs, fake, tmp_path, review, **kw)
    state = load_state(tmp_path / ".runpod" / "state.yaml")
    assert state["endpoints"][NAME]["client_data"] is False
    assert "approved_by" not in load_releases(tmp_path / "releases.yaml")[-1]


def test_promote_refused_with_pending_parity(configs, tmp_path, deployed):  # noqa: F811
    _refused_promote(configs, deployed, tmp_path, "parity")


def test_promote_refused_with_failed_parity(configs, tmp_path, deployed):  # noqa: F811
    _set_parity(tmp_path, "fail")
    _refused_promote(configs, deployed, tmp_path, "parity")


def test_promote_refused_without_review_document(configs, tmp_path, deployed):  # noqa: F811
    _set_parity(tmp_path)
    _refused_promote(configs, deployed, tmp_path, "missing.md", review=tmp_path / "missing.md")


def test_promote_refused_when_review_not_approved(configs, tmp_path, deployed):  # noqa: F811
    _set_parity(tmp_path)
    review = _review(tmp_path / "r.md", decision="rejected")
    _refused_promote(configs, deployed, tmp_path, "approved", review=review)


def test_promote_refused_when_review_is_older_than_12_months(configs, tmp_path, deployed):  # noqa: F811
    _set_parity(tmp_path)
    review = _review(tmp_path / "r.md", on="2025-10-01")
    _refused_promote(configs, deployed, tmp_path, "12 months", review=review)


def test_promote_refused_for_undeployed_model(configs, tmp_path):  # noqa: F811
    with pytest.raises(manage.ManageRefused, match="deploy"):
        _promote(configs, FakeRunpod(), tmp_path, _review(tmp_path / "r.md"))


def test_promote_queue_endpoint_needs_stated_retention(configs, tmp_path):  # noqa: F811
    _edit(configs, kind="queue")
    fake = FakeRunpod()
    _deploy(configs, fake, tmp_path)
    _set_parity(tmp_path)
    _refused_promote(configs, fake, tmp_path, "retention")
    _promote(configs, fake, tmp_path, _review(tmp_path / "r.md"), retention="24 h job TTL")
    assert load_releases(tmp_path / "releases.yaml")[-1]["retention"] == "24 h job TTL"


# --- deploy --profile --------------------------------------------------------------------------


def test_profile_office_refused_for_unpromoted_model(configs, tmp_path, deployed):  # noqa: F811
    n = len(deployed.writes)
    with pytest.raises(dp.DeployRefused, match="promoted"):
        _deploy(configs, deployed, tmp_path, profile="office")
    assert len(deployed.writes) == n


def test_profile_office_then_off_for_promoted_model(configs, tmp_path, deployed):  # noqa: F811
    _set_parity(tmp_path)
    _promote(configs, deployed, tmp_path, _review(tmp_path / "r.md"))
    res = _deploy(configs, deployed, tmp_path, profile="office")
    assert res.action == "patched"
    assert deployed.endpoints[0]["workers"]["min"] == 1
    _deploy(configs, deployed, tmp_path, profile="off")
    assert deployed.endpoints[0]["workers"]["min"] == 0
    assert len(load_releases(tmp_path / "releases.yaml")) == 1


def test_profile_off_allowed_for_unpromoted_model(configs, tmp_path, deployed):  # noqa: F811
    res = _deploy(configs, deployed, tmp_path, profile="off")
    assert res.action == "up_to_date"


# --- the CLI -----------------------------------------------------------------------------------

runner = CliRunner()


def _cli(configs, tmp_path, fake, monkeypatch, *args):  # noqa: F811
    monkeypatch.setattr(runpod_cli, "make_api", lambda: _api(fake))
    return runner.invoke(
        app,
        [
            "runpod",
            args[0],
            "dotsocr",
            "--configs-dir",
            str(configs),
            "--state",
            str(tmp_path / ".runpod" / "state.yaml"),
            "--releases",
            str(tmp_path / "releases.yaml"),
            *args[1:],
        ],
    )


def test_cli_status_masks_id(configs, tmp_path, deployed, monkeypatch):  # noqa: F811
    result = _cli(configs, tmp_path, deployed, monkeypatch, "status")
    assert result.exit_code == 0, result.output
    assert NAME in result.output and "pending" in result.output
    assert deployed.endpoints[0]["id"] not in result.output


def test_cli_rollback_then_promote(configs, tmp_path, deployed, monkeypatch):  # noqa: F811
    _deploy(configs, deployed, tmp_path, digest=DIGEST2)
    result = _cli(configs, tmp_path, deployed, monkeypatch, "rollback")
    assert result.exit_code == 0, result.output
    assert "rolled back" in result.output
    _set_parity(tmp_path)
    review = str(_review(tmp_path / "r.md"))
    result = _cli(
        configs, tmp_path, deployed, monkeypatch,
        "deploy", "--promote", "--review", review, "--approver", "boss",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert load_state(tmp_path / ".runpod" / "state.yaml")["endpoints"][NAME]["client_data"]


def test_cli_promote_refusal_exits_2(configs, tmp_path, deployed, monkeypatch):  # noqa: F811
    review = str(_review(tmp_path / "r.md"))
    result = _cli(
        configs, tmp_path, deployed, monkeypatch, "deploy", "--promote", "--review", review
    )
    assert result.exit_code == 2
    assert "parity" in result.output
