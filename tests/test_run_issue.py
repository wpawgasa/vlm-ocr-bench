"""Tests for the issue runner: scripts/run_issue.sh and scripts/run_issue/lib.sh.

Unit tests source lib.sh in bash. End-to-end tests run the real runner against a
throwaway git repo whose `origin` is a local bare repo, with fake `gh`, `claude`
and `uv` (tests/fixtures/run_issue/fake_tools.py) on PATH.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_issue.sh"
LIB = ROOT / "scripts" / "run_issue" / "lib.sh"
TEMPLATES = ROOT / "scripts" / "run_issue"
FAKE_TOOLS = ROOT / "tests" / "fixtures" / "run_issue" / "fake_tools.py"
REPO = "o/r"


def call(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run one lib.sh function: call("parse_tier", title, body)."""
    return subprocess.run(
        ["bash", "-c", 'source "$0" && "$@"', str(LIB), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def out(*args: str, env: dict | None = None) -> str:
    result = call(*args, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.rstrip("\n")


# --- tiers and routing -------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "body", "tier"),
    [
        ("runpod-serving 1.1 [Sonnet]: config loader", "", "Sonnet"),
        ("docparse 3.2 [Opus]: merge", "", "Opus"),
        ("x [Human]: label data", "", "Human"),
        ("x [Fable]: draft spec", "", "Fable"),
        ("Fix the mapper", "Intro\n\n**Tier:** [Opus]\n", "Opus"),
        ("Fix the mapper", "Intro\r\n**Tier:** [Sonnet]\r\n", "Sonnet"),
        ("runpod-serving R1: group", "- [ ] 1.1 [Sonnet] Add config (#117)\n", ""),
        ("Fix the mapper", "No tag here.", ""),
    ],
)
def test_parse_tier(title: str, body: str, tier: str) -> None:
    assert out("parse_tier", title, body) == tier


def test_strip_tier() -> None:
    assert out("strip_tier", "runpod-serving 1.1 [Sonnet]: config loader") == (
        "runpod-serving 1.1: config loader"
    )
    assert out("strip_tier", "Fix the mapper") == "Fix the mapper"


def test_slugify() -> None:
    assert out("slugify", "runpod-serving 1.1: Runpod model config (ocr_bench/runpod)") == (
        "runpod-serving-1-1-runpod-model-config-o"
    )
    assert len(out("slugify", "a" * 100)) == 40
    assert out("slugify", "ทดสอบ !!!") == ""


def test_branch_name() -> None:
    assert out("branch_name", "117", "runpod-serving 1.1 [Sonnet]: Config") == (
        "issue/117-runpod-serving-1-1-config"
    )


def test_branch_name_falls_back_for_non_ascii_title() -> None:
    assert out("branch_name", "42", "แก้ไขวันที่ [Opus]") == "issue/42-issue"


@pytest.mark.parametrize(
    ("tier", "allow", "expected"),
    [
        ("Sonnet", "0", "sonnet\t"),
        ("Opus", "0", "opus\t"),
        ("Human", "0", "skip\tneeds a human"),
        ("Fable", "1", "fable\t"),
        ("Fable", "0", "opus\tFable downgraded to Opus (no --allow-fable)"),
    ],
)
def test_resolve_model(tier: str, allow: str, expected: str) -> None:
    assert out("resolve_model", tier, allow) == expected


def test_commit_trailer() -> None:
    assert out("commit_trailer", "opus").startswith("Co-Authored-By: Claude Opus")
    assert out("commit_trailer", "sonnet").startswith("Co-Authored-By: Claude Sonnet")
    assert out("commit_trailer", "fable").startswith("Co-Authored-By: Claude Fable")


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("All good.\n\nRESULT: done", "done"),
        ("RESULT: blocked: needs an H100\n", "blocked: needs an H100"),
        ("RESULT: blocked: x\nretried\nRESULT: done\n", "done"),
        ("I finished.", "missing"),
        ("", "missing"),
    ],
)
def test_result_status(text: str, status: str) -> None:
    assert out("result_status", text) == status


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('Thinking...\n{"tier": "Opus", "reason": "crosses modules"}', "Opus\tcrosses modules"),
        ('{"tier": "sonnet", "reason": "routine\\ttests"}', "Sonnet\troutine tests"),
        (
            '{"tier": "Haiku", "reason": "cheap"}',
            "Sonnet\trouting output unparseable; model-routing default",
        ),
        ("no json at all", "Sonnet\trouting output unparseable; model-routing default"),
    ],
)
def test_parse_route_result(text: str, expected: str) -> None:
    assert out("parse_route_result", text) == expected


# --- files -------------------------------------------------------------------

PR_BODY_OK = (
    "## Summary\nx\n\n## Implementation details\nx\n\n## In plain words\nx\n\n"
    "## Verification\nx\n\n## Routing\nx\n"
)


def test_check_pr_body_accepts_complete_body(tmp_path: Path) -> None:
    body = tmp_path / "PR_BODY.md"
    body.write_text(PR_BODY_OK)
    assert call("check_pr_body", str(body)).returncode == 0


def test_check_pr_body_names_missing_headings(tmp_path: Path) -> None:
    body = tmp_path / "PR_BODY.md"
    body.write_text(PR_BODY_OK.replace("## In plain words", "## Plain"))
    result = call("check_pr_body", str(body))
    assert result.returncode == 1
    assert "PR body missing heading: ## In plain words" in result.stdout


def test_check_pr_body_reports_missing_file(tmp_path: Path) -> None:
    result = call("check_pr_body", str(tmp_path / "nope.md"))
    assert result.returncode == 1
    assert "PR body file missing" in result.stdout


def test_tick_parent_line_ticks_only_the_exact_ref(tmp_path: Path) -> None:
    body = tmp_path / "body.md"
    body.write_text(
        "### Tasks\r\n"
        "- [ ] 1.10 [Sonnet] other (#1170)\r\n"
        "- [ ] 1.1 [Sonnet] this one (#117)\r\n"
        "- [ ] 1.2 [Sonnet] next (#118)\r\n",
        newline="",
    )
    assert call("tick_parent_line", str(body), "117").returncode == 0
    assert body.read_bytes().decode() == (
        "### Tasks\r\n"
        "- [ ] 1.10 [Sonnet] other (#1170)\r\n"
        "- [x] 1.1 [Sonnet] this one (#117)\r\n"
        "- [ ] 1.2 [Sonnet] next (#118)\r\n"
    )


def test_tick_parent_line_leaves_file_alone_without_unchecked_match(tmp_path: Path) -> None:
    body = tmp_path / "body.md"
    original = "- [x] 1.1 done (#117)\n- [ ] 1.2 other (#118)\n"
    body.write_text(original)
    assert call("tick_parent_line", str(body), "117").returncode == 1
    assert body.read_text() == original


def test_render_template_is_literal_and_single_pass(tmp_path: Path) -> None:
    template = tmp_path / "t.md"
    template.write_text("T={{ISSUE_TITLE}}\nB={{ISSUE_BODY}}\nU={{UNKNOWN}}\n")
    nasty = "run $(rm -rf /) `id` & \\n {{ISSUE_TITLE}} ทดสอบ"
    rendered = out("render_template", str(template), "ISSUE_TITLE=a=b", f"ISSUE_BODY={nasty}")
    assert rendered == f"T=a=b\nB={nasty}\nU={{{{UNKNOWN}}}}"


def test_json_field(tmp_path: Path) -> None:
    data = tmp_path / "out.json"
    data.write_text('{"result": "line1\\nRESULT: done", "total_cost_usd": 0.5, "num_turns": 7}')
    assert out("json_field", str(data), "result") == "line1\nRESULT: done"
    assert out("json_field", str(data), "num_turns", "0") == "7"
    assert out("json_field", str(data), "missing", "0") == "0"
    assert out("json_field", str(tmp_path / "absent.json"), "result", "x") == "x"
    (tmp_path / "bad.json").write_text("not json")
    assert out("json_field", str(tmp_path / "bad.json"), "num_turns", "0") == "0"


def test_add_num() -> None:
    assert out("add_num", "0.25", "1.5") == "1.7500"
    assert out("add_num", "0", "") == "0.0000"


def test_load_tools_skips_comments_and_blanks(tmp_path: Path) -> None:
    tools = tmp_path / "tools.txt"
    tools.write_text("# comment\nRead\n\n  \nBash(uv run:*)\n")
    assert out("load_tools", str(tools)).splitlines() == ["Read", "Bash(uv run:*)"]


# --- fakes -------------------------------------------------------------------


class Fakes:
    """Fake gh/claude/uv on PATH, driven by a scenario file (see fake_tools.py)."""

    def __init__(self, tmp_path: Path) -> None:
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.log = tmp_path / "fake-log.jsonl"
        self.state = tmp_path / "fake-state.json"
        self.scenario_path = tmp_path / "scenario.json"
        self.scenario: dict = {"gh": [], "uv": [], "claude": {}}
        for tool in ("gh", "claude", "uv"):
            wrapper = self.bin / tool
            wrapper.write_text(f'#!/usr/bin/env bash\nexec python3 "{FAKE_TOOLS}" {tool} "$@"\n')
            wrapper.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_LOG": str(self.log),
            "FAKE_STATE": str(self.state),
            "FAKE_SCENARIO": str(self.scenario_path),
        }
        self._save()

    def _save(self) -> None:
        self.scenario_path.write_text(json.dumps(self.scenario))

    def _rule(self, tool: str, match: str, responses: tuple, first: bool) -> None:
        rule = {
            "match": match,
            "responses": [r if isinstance(r, dict) else {"stdout": r} for r in responses] or [{}],
        }
        if first:
            self.scenario[tool].insert(0, rule)
        else:
            self.scenario[tool].append(rule)
        self._save()

    def gh(self, match: str, *responses: str | dict, first: bool = False) -> None:
        self._rule("gh", match, responses, first)

    def uv(self, match: str, *responses: str | dict, first: bool = False) -> None:
        self._rule("uv", match, responses, first)

    def claude(self, kind: str, *actions: dict) -> None:
        self.scenario["claude"][kind] = list(actions)
        self._save()

    def calls(self, tool: str) -> list[dict]:
        if not self.log.exists():
            return []
        entries = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [e for e in entries if e["tool"] == tool]

    def gh_calls(self, prefix: str) -> list[dict]:
        return [c for c in self.calls("gh") if " ".join(c["argv"]).startswith(prefix)]

    def reset(self) -> None:
        """Forget logged calls and response counters (between two runs)."""
        self.log.unlink(missing_ok=True)
        self.state.unlink(missing_ok=True)


def add_issue(
    fakes: Fakes,
    n: int,
    title: str,
    body: str = "Do the thing.",
    state: str = "open",
    parent: int | None = None,
    subs: list[tuple[int, str]] | None = None,
) -> None:
    """Script the gh replies describing issue N."""
    fakes.gh(
        rf"^api repos/{REPO}/issues/{n}/sub_issues ",
        "".join(f"{s} {st}\n" for s, st in subs or []),
    )
    fakes.gh(rf"^issue view {n} -R {REPO} --json title ", title + "\n")
    fakes.gh(rf"^issue view {n} -R {REPO} --json body ", body + "\n")
    fakes.gh(rf"^issue view {n} -R {REPO} --json state ", state + "\n")
    fakes.gh(
        rf"^api repos/{REPO}/issues/{n}/parent ",
        f"{parent}\n" if parent else {"exit": 1},
    )


@pytest.fixture
def fakes(tmp_path: Path) -> Fakes:
    return Fakes(tmp_path)


# --- issue expansion ---------------------------------------------------------


def test_expand_issues_replaces_group_with_sub_issues(fakes: Fakes) -> None:
    add_issue(fakes, 116, "group R1", subs=[(117, "open"), (118, "closed"), (119, "open")])
    add_issue(fakes, 120, "leaf [Opus]: thing")
    add_issue(fakes, 117, "leaf [Sonnet]: dup")
    lines = out("expand_issues", REPO, "116", "120", "117", env=fakes.env).splitlines()
    assert lines == ["117 open", "118 closed", "119 open", "120 open"]


def test_expand_issues_falls_back_to_task_refs(fakes: Fakes) -> None:
    body = (
        "Intro mentions (#5).\n\n### Tasks\n"
        "- [ ] 1.1 [Sonnet] first (#11)\n"
        "- [x] 1.2 [Sonnet] second (#12)\n\n"
        "### Notes\n- [ ] unrelated (#99)\n"
    )
    add_issue(fakes, 10, "group", body=body)
    add_issue(fakes, 11, "a [Sonnet]: a")
    add_issue(fakes, 12, "b [Sonnet]: b", state="closed")
    lines = out("expand_issues", REPO, "10", env=fakes.env).splitlines()
    assert lines == ["11 open", "12 closed"]


def test_expand_issues_keeps_leaf(fakes: Fakes) -> None:
    add_issue(fakes, 7, "leaf [Opus]: x", state="closed")
    assert out("expand_issues", REPO, "7", env=fakes.env) == "7 closed"


# --- templates and allowlist -------------------------------------------------

TEMPLATE_KEYS = {
    "implement.md": [
        "REPO",
        "BASE",
        "ISSUE_NUMBER",
        "ISSUE_TITLE",
        "ROUTING",
        "TRAILER",
        "ISSUE_BODY",
    ],
    "fix.md": ["ISSUE_NUMBER", "ISSUE_TITLE", "FAILURE_KIND", "TRAILER", "FAILURE_LOG"],
    "route.md": ["ISSUE_NUMBER", "ISSUE_TITLE", "ISSUE_BODY"],
}


@pytest.mark.parametrize("name", sorted(TEMPLATE_KEYS))
def test_template_renders_every_placeholder(name: str) -> None:
    pairs = [f"{key}=<{key.lower()}>" for key in TEMPLATE_KEYS[name]]
    rendered = out("render_template", str(TEMPLATES / name), *pairs)
    assert "{{" not in rendered
    assert f"<!-- run_issue:{name.removesuffix('.md')} -->" in rendered
    for key in TEMPLATE_KEYS[name]:
        assert f"<{key.lower()}>" in rendered


def test_implement_prompt_requires_superpowers_tests_and_pr_body() -> None:
    text = (TEMPLATES / "implement.md").read_text()
    for needle in [
        "superpowers:test-driven-development",
        "superpowers:systematic-debugging",
        "superpowers:verification-before-completion",
        'uv run pytest -m "not live"',
        ".run_issue/PR_BODY.md",
        "## Summary",
        "## Implementation details",
        "## In plain words",
        "## Verification",
        "## Routing",
        "RESULT: done",
        "RESULT: blocked:",
        "Closes #{{ISSUE_NUMBER}}",
    ]:
        assert needle in text, needle


def test_fix_prompt_uses_systematic_debugging() -> None:
    text = (TEMPLATES / "fix.md").read_text()
    assert "superpowers:systematic-debugging" in text
    assert "RESULT: done" in text


def test_route_prompt_uses_model_routing_and_asks_for_json() -> None:
    text = (TEMPLATES / "route.md").read_text()
    assert "model-routing" in text
    assert '{"tier":' in text


def test_allowlist_excludes_irreversible_commands() -> None:
    tools = out("load_tools", str(TEMPLATES / "allowed_tools.txt")).splitlines()
    assert {"Read", "Edit", "Write", "Skill", "Bash(uv run:*)", "Bash(git commit:*)"} <= set(tools)
    for entry in tools:
        assert not entry.startswith(("Bash(git push", "Bash(gh pr", "Bash(gh api")), entry
