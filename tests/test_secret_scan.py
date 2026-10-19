"""The gitleaks secret scan (design D9 "Keys", task 5.5): config rules, hook and CI step."""

import re
import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
FAKE_KEY = "rpa_" + "X" * 24
FAKE_TEST_HOST = "abcdefghij12.api.runpod.ai"
# Built at runtime so this file does not itself trip the scan it tests.
REALISTIC_HOST = "q7w8e9r0t1y2u3" + ".api.runpod.ai"


def _config() -> dict:
    return tomllib.loads((REPO / ".gitleaks.toml").read_text())


def _rule(rule_id: str) -> dict:
    return next(r for r in _config()["rules"] if r["id"] == rule_id)


def test_config_extends_the_default_rules():
    assert _config()["extend"]["useDefault"] is True


def test_runpod_key_rule_matches_a_key():
    rule = _rule("runpod-api-key")
    assert re.search(rule["regex"], f"RUNPOD_API_KEY={FAKE_KEY}")
    assert not re.search(rule["regex"], "rpa_short")


def test_runpod_host_rule_matches_an_endpoint_host_but_allows_the_test_host():
    rule = _rule("runpod-endpoint-host")
    assert re.search(rule["regex"], f"https://{REALISTIC_HOST}/v1/models")
    allowed = [rx for al in rule["allowlists"] for rx in al["regexes"]]
    assert any(re.search(rx, FAKE_TEST_HOST) for rx in allowed)
    assert not any(re.search(rx, REALISTIC_HOST) for rx in allowed)


def test_pre_commit_runs_gitleaks():
    cfg = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text())
    hooks = [h["id"] for repo in cfg["repos"] for h in repo["hooks"]]
    assert "gitleaks" in hooks


def test_ci_runs_gitleaks_over_the_history():
    ci = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
    steps = ci["jobs"]["secret-scan"]["steps"]
    assert steps[0]["with"]["fetch-depth"] == 0
    run = "\n".join(s.get("run", "") for s in steps)
    assert "sha256sum -c" in run
    assert "gitleaks git" in run
