"""Tests for the issue runner: scripts/run_issue.sh and scripts/run_issue/lib.sh.

Unit tests source lib.sh in bash. End-to-end tests run the real runner against a
throwaway git repo whose `origin` is a local bare repo, with fake `gh`, `claude`
and `uv` (tests/fixtures/run_issue/fake_tools.py) on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
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


# --- end to end: sandbox -----------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path, fakes: Fakes) -> Fakes:
    """A work repo whose origin is a local bare repo, plus the fakes."""
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text("")
    fakes.env.update(
        GIT_CONFIG_GLOBAL=str(gitconfig),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
        RUN_ISSUE_CHECKS_WAIT_SECS="0",
        RUN_ISSUE_CHECKS_POLL_SECS="0",
        ANTHROPIC_API_KEY="sk-must-not-reach-claude",
        FAKE_ORIGIN=str(origin),
    )

    def git(*args: str, cwd: Path = work) -> None:
        subprocess.run(["git", *args], cwd=cwd, env=fakes.env, check=True, capture_output=True)

    git("init", "-q", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    work.mkdir()
    git("init", "-q", "-b", "main")
    (work / "README.md").write_text("sandbox\n")
    git("add", "README.md")
    git("commit", "-qm", "init")
    git("remote", "add", "origin", str(origin))
    git("push", "-q", "-u", "origin", "main")
    fakes.work, fakes.origin = work, origin
    fakes.claude("implement", {"commit": "test", "pr_body": True})
    return fakes


def run_runner(sandbox: Fakes, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RUNNER), "--repo", REPO, *args],
        cwd=sandbox.work,
        env=sandbox.env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def claude_calls(sandbox: Fakes, kind: str | None = None) -> list[dict]:
    return [c for c in sandbox.calls("claude") if kind is None or c["kind"] == kind]


def arg_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# --- end to end: CLI and routing ---------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--issues", "abc"],
        ["--issues", "1,"],
        ["--issues", "1", "--max-ci-fixes", "x"],
        ["--bogus"],
        ["--issues"],
    ],
)
def test_usage_errors_exit_2(sandbox: Fakes, args: list[str]) -> None:
    result = run_runner(sandbox, *args)
    assert result.returncode == 2, result.stderr
    assert "Usage:" in result.stderr


def test_unauthenticated_gh_exits_2(sandbox: Fakes) -> None:
    sandbox.gh(r"^auth status", {"exit": 1}, first=True)
    result = run_runner(sandbox, "--issues", "1")
    assert result.returncode == 2
    assert "gh is not authenticated" in result.stderr


def test_dry_run_plans_routes_and_skips(sandbox: Fakes) -> None:
    add_issue(
        sandbox,
        116,
        "runpod-serving R1: group",
        subs=[(117, "open"), (118, "closed"), (119, "open")],
    )
    add_issue(sandbox, 117, "runpod-serving 1.1 [Sonnet]: config loader")
    add_issue(sandbox, 118, "runpod-serving 1.2 [Sonnet]: configs")
    add_issue(sandbox, 119, "Untagged parity set")
    add_issue(sandbox, 120, "runpod-serving 5.4 [Human]: pin weights")
    sandbox.claude("route", {"result": 'ok\n{"tier": "Opus", "reason": "crosses modules"}'})

    result = run_runner(sandbox, "--issues", "116,120", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "-> sonnet (tier tag [Sonnet] on the issue)" in result.stdout
    assert "-> opus (model-routing skill: crosses modules)" in result.stdout
    summary = result.stdout[result.stdout.index("ISSUE") :]
    assert "#117" in summary and "planned" in summary
    assert "#118" in summary and "skipped-closed" in summary
    assert "#120" in summary and "skipped-human" in summary
    calls = claude_calls(sandbox)
    assert [c["kind"] for c in calls] == ["route"]
    route = calls[0]
    assert arg_after(route["argv"], "--model") == "sonnet"
    tools = route["argv"][route["argv"].index("--allowedTools") + 1 :]
    assert tools == ["Read", "Glob", "Grep", "Skill"]
    assert route["api_key_set"] is False
    assert "Untagged parity set" in route["prompt"]
    assert not list((sandbox.work / ".claude" / "worktrees").glob("issue-*"))
    assert not sandbox.gh_calls("pr ")


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (
            [],
            "-> opus (tier tag [Fable] on the issue; Fable downgraded to Opus (no --allow-fable))",
        ),
        (["--allow-fable"], "-> fable"),
    ],
)
def test_fable_gate(sandbox: Fakes, flags: list[str], expected: str) -> None:
    add_issue(sandbox, 5, "spec 0.1 [Fable]: draft design")
    result = run_runner(sandbox, "--issues", "5", "--dry-run", *flags)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


# --- end to end: implement phase ---------------------------------------------


def ship_ok(sandbox: Fakes, n: int) -> None:
    """Script a PR for issue N whose CI passes."""
    sandbox.gh(rf"^pr list -R {REPO} --head issue/{n}-", "")
    sandbox.gh(rf"^pr create .*--head issue/{n}-", f"https://github.com/{REPO}/pull/{n}\n")
    sandbox.gh(
        rf"^pr view \S+/pull/{n} .*headRefOid",
        "@SH:git --git-dir=\"$FAKE_ORIGIN\" for-each-ref --format='%(objectname)' "
        f"'refs/heads/issue/{n}-*'",
    )
    sandbox.gh(rf"^pr checks \S+/pull/{n} -R {REPO} --json", "1\n")
    sandbox.gh(rf"^pr checks \S+/pull/{n} -R {REPO} --watch", "")


def test_implement_call_uses_model_allowlist_and_worktree(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "runpod-serving 1.1 [Sonnet]: config loader")
    ship_ok(sandbox, 117)

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 0, result.stderr
    (impl,) = claude_calls(sandbox, "implement")
    assert arg_after(impl["argv"], "--model") == "sonnet"
    assert arg_after(impl["argv"], "--permission-mode") == "acceptEdits"
    assert "--dangerously-skip-permissions" not in impl["argv"]
    tools = impl["argv"][impl["argv"].index("--allowedTools") + 1 :]
    assert "Bash(uv run:*)" in tools and not any("git push" in t for t in tools)
    assert impl["api_key_set"] is False
    assert impl["cwd"] == str(sandbox.work / ".claude" / "worktrees" / "issue-117")
    assert "Closes #117" in impl["prompt"]
    assert "tier tag [Sonnet] on the issue" in impl["prompt"]
    assert "Co-Authored-By: Claude Sonnet" in impl["prompt"]
    assert [c["argv"][:1] for c in sandbox.calls("uv")][0] == ["sync"]


def test_issue_body_reaches_prompt_verbatim(sandbox: Fakes) -> None:
    body = "Run `id` and $(touch pwned) & keep {{ISSUE_TITLE}} \\ ทดสอบ"
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y", body=body)
    ship_ok(sandbox, 117)
    run_runner(sandbox, "--issues", "117")
    (impl,) = claude_calls(sandbox, "implement")
    assert body in impl["prompt"]
    assert not (sandbox.work / "pwned").exists()


def test_blocked_agent_halts_without_fix_rounds(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y")
    add_issue(sandbox, 118, "x 1.2 [Sonnet]: z")
    sandbox.claude("implement", {"result": "Cannot.\nRESULT: blocked: needs an H100"})

    result = run_runner(sandbox, "--issues", "117,118")

    assert result.returncode == 1
    assert [c["kind"] for c in claude_calls(sandbox)] == ["implement"]
    assert "blocked" in result.stdout
    assert not (sandbox.work / ".claude" / "worktrees" / "issue-118").exists()
    assert (sandbox.work / ".claude" / "worktrees" / "issue-117").exists()


@pytest.mark.parametrize(
    ("first_attempt", "expected_failure"),
    [
        ({"commit": "code", "pr_body": True}, "No files under tests/ changed"),
        ({"commit": "test", "pr_body": True, "dirty": True}, "Uncommitted changes"),
        ({"commit": "test"}, "PR body file missing"),
        ({"pr_body": True}, "No commits on the branch"),
        (
            {"commit": "test", "pr_body": True, "result": "I think it is fine."},
            "did not end with 'RESULT: done'",
        ),
    ],
)
def test_failed_post_check_sends_fix_prompt(
    sandbox: Fakes, first_attempt: dict, expected_failure: str
) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y")
    ship_ok(sandbox, 117)
    sandbox.claude("implement", first_attempt)
    sandbox.claude("fix", {"commit": "test", "pr_body": True})

    result = run_runner(sandbox, "--issues", "117")

    fixes = claude_calls(sandbox, "fix")
    assert fixes, result.stderr
    assert expected_failure in fixes[0]["prompt"]
    assert "local verification" in fixes[0]["prompt"]


@pytest.mark.parametrize(
    ("tier", "fix_models"), [("Sonnet", ["sonnet", "opus"]), ("Opus", ["opus"])]
)
def test_local_verification_failure_escalates_then_halts(
    sandbox: Fakes, tier: str, fix_models: list[str]
) -> None:
    add_issue(sandbox, 117, f"x 1.1 [{tier}]: y")
    sandbox.uv(r"^run ruff check", {"stdout": "E501 line too long\n", "exit": 1})
    sandbox.claude("fix", {"commit": "test", "pr_body": True})

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 1
    fixes = claude_calls(sandbox, "fix")
    assert [arg_after(c["argv"], "--model") for c in fixes] == fix_models
    assert "E501 line too long" in fixes[0]["prompt"]
    assert not sandbox.gh_calls("pr create")
    assert "blocked" in result.stdout


def test_sonnet_escalates_to_opus_and_recovers(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y")
    ship_ok(sandbox, 117)
    sandbox.uv(r"^run ruff check", {"exit": 1}, {"exit": 1}, {"exit": 0})
    sandbox.claude("fix", {"commit": "test", "pr_body": True})

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 0, result.stderr
    fixes = claude_calls(sandbox, "fix")
    assert [arg_after(c["argv"], "--model") for c in fixes] == ["sonnet", "opus"]
    assert "sonnet>opus" in result.stdout


def test_existing_non_worktree_directory_is_refused(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y")
    stray = sandbox.work / ".claude" / "worktrees" / "issue-117"
    stray.mkdir(parents=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=sandbox.work, capture_output=True, text=True, check=True
    ).stdout

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 1
    assert not claude_calls(sandbox, "implement")
    assert "not a git worktree" in result.stderr + result.stdout
    new_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=sandbox.work, capture_output=True, text=True, check=True
    ).stdout
    assert new_head == head


def test_resume_after_retitle_adopts_existing_branch(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: old title")
    ship_ok(sandbox, 117)
    sandbox.gh(rf"^pr checks \S+/pull/117 -R {REPO} --watch", {"exit": 1}, first=True)
    first = run_runner(sandbox, "--issues", "117", "--max-ci-fixes", "0")
    assert first.returncode == 1  # CI red: the worktree is kept for the resume
    sandbox.reset()
    sandbox.scenario["gh"] = []
    sandbox._save()
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: brand new title")
    ship_ok(sandbox, 117)

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 0, result.stderr
    assert not claude_calls(sandbox, "implement")
    assert "issue/117-x-1-1-old-title" in result.stderr + result.stdout


# --- end to end: ship phase --------------------------------------------------


def origin_has_branch(sandbox: Fakes, n: int) -> bool:
    refs = subprocess.run(
        [
            "git",
            f"--git-dir={sandbox.origin}",
            "for-each-ref",
            "--format=%(refname)",
            f"refs/heads/issue/{n}-*",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return bool(refs.strip())


def test_happy_path_expands_group_and_merges(sandbox: Fakes) -> None:
    parent_body = "### Tasks\n- [ ] 1.1 [Sonnet] loader (#117)\n- [ ] 1.2 [Sonnet] configs (#118)\n"
    add_issue(
        sandbox,
        116,
        "runpod-serving R1: group",
        body=parent_body,
        subs=[(117, "open"), (118, "closed")],
    )
    add_issue(sandbox, 117, "runpod-serving 1.1 [Sonnet]: config loader", parent=116)

    ship_ok(sandbox, 117)

    result = run_runner(sandbox, "--issues", "116")

    assert result.returncode == 0, result.stderr
    assert origin_has_branch(sandbox, 117)
    (create,) = sandbox.gh_calls("pr create")
    assert arg_after(create["argv"], "--title") == "runpod-serving 1.1: config loader (#117)"
    assert arg_after(create["argv"], "--base") == "main"
    assert "## In plain words" in create["body"]
    assert "Generated with [Claude Code](https://claude.com/claude-code)" in create["body"]
    (merge,) = sandbox.gh_calls("pr merge")
    pushed = subprocess.run(
        [
            "git",
            f"--git-dir={sandbox.origin}",
            "for-each-ref",
            "--format=%(objectname)",
            "refs/heads/issue/117-*",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert merge["argv"][3:] == [
        "-R",
        REPO,
        "--squash",
        "--delete-branch",
        "--match-head-commit",
        pushed,
    ]
    (edit,) = sandbox.gh_calls("issue edit 116")
    assert "- [x] 1.1 [Sonnet] loader (#117)" in edit["body"]
    assert "- [ ] 1.2 [Sonnet] configs (#118)" in edit["body"]
    assert sandbox.gh_calls("issue comment 117")
    assert not (sandbox.work / ".claude" / "worktrees" / "issue-117").exists()
    summary = result.stdout[result.stdout.index("ISSUE") :]
    assert "merged" in summary and f"https://github.com/{REPO}/pull/117" in summary
    assert "0.2500" in summary


def test_ci_failure_gets_one_fix_then_merges(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y")
    ship_ok(sandbox, 117)
    sandbox.gh(rf"^pr checks \S+/pull/117 -R {REPO} --watch", {"exit": 1}, {"exit": 0}, first=True)
    sandbox.gh(r"^run list .*--commit", "555\n")
    sandbox.gh(r"^run view 555", "FAILED tests/test_x.py::test_y - assert 1 == 2\n")
    sandbox.claude("fix", {"commit": "test", "pr_body": True})

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 0, result.stderr
    (fix,) = claude_calls(sandbox, "fix")
    assert "FAILED tests/test_x.py::test_y" in fix["prompt"]
    assert "CI (GitHub Actions)" in fix["prompt"]
    assert arg_after(fix["argv"], "--model") == "sonnet"
    assert len(sandbox.gh_calls("pr merge")) == 1
    assert sandbox.gh_calls("pr edit")


def test_ci_failures_exhaust_fix_rounds_and_halt(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y")
    add_issue(sandbox, 118, "x 1.2 [Sonnet]: z")
    ship_ok(sandbox, 117)
    sandbox.gh(rf"^pr checks \S+/pull/117 -R {REPO} --watch", {"exit": 1}, first=True)
    sandbox.gh(r"^run list .*--commit", "555\n")
    sandbox.gh(r"^run view 555", "boom\n")
    sandbox.claude("fix", {"commit": "test", "pr_body": True})

    result = run_runner(sandbox, "--issues", "117,118", "--max-ci-fixes", "1")

    assert result.returncode == 1
    assert len(claude_calls(sandbox, "fix")) == 1
    assert not sandbox.gh_calls("pr merge")
    (comment,) = sandbox.gh_calls("pr comment")
    assert "not merged" in " ".join(comment["argv"])
    assert "ci-failed" in result.stdout
    assert not (sandbox.work / ".claude" / "worktrees" / "issue-118").exists()


def test_no_checks_never_merges(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y")
    ship_ok(sandbox, 117)
    sandbox.gh(rf"^pr checks \S+/pull/117 -R {REPO} --json", "0\n", first=True)

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 1
    assert not sandbox.gh_calls("pr merge")
    assert "no CI checks" in result.stderr


def test_head_sha_mismatch_never_merges(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y")
    ship_ok(sandbox, 117)
    sandbox.gh(r"^pr view \S+/pull/117 .*headRefOid", "deadbeef\n", first=True)

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 1
    assert not sandbox.gh_calls("pr merge")
    assert not sandbox.gh_calls("pr checks https://github.com/o/r/pull/117 -R o/r --watch")


def test_rerun_reuses_worktree_and_pr(sandbox: Fakes) -> None:
    add_issue(sandbox, 117, "x 1.1 [Sonnet]: y")
    ship_ok(sandbox, 117)
    sandbox.gh(rf"^pr checks \S+/pull/117 -R {REPO} --watch", {"exit": 1}, first=True)
    assert run_runner(sandbox, "--issues", "117", "--max-ci-fixes", "0").returncode == 1

    sandbox.reset()
    sandbox.scenario["gh"].pop(0)  # CI is green now
    sandbox.gh(
        rf"^pr list -R {REPO} --head issue/117-",
        f"https://github.com/{REPO}/pull/117\n",
        first=True,
    )

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 0, result.stderr
    assert not claude_calls(sandbox)
    assert not sandbox.gh_calls("pr create")
    assert sandbox.gh_calls("pr edit")
    assert len(sandbox.gh_calls("pr merge")) == 1


# --- lint --------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_clean() -> None:
    result = subprocess.run(
        ["shellcheck", "-x", "scripts/run_issue.sh", "scripts/run_issue/lib.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
