"""Tests for the issue runner: scripts/run_issue.sh and scripts/run_issue/lib.sh.

Unit tests source lib.sh in bash. End-to-end tests run the real runner against a
throwaway git repo whose `origin` is a local bare repo, with fake `gh`, `claude`
and `uv` (tests/fixtures/run_issue/fake_tools.py) on PATH.
"""

from __future__ import annotations

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
