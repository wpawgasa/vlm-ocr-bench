# Issue Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `scripts/run_issue.sh --issues 1,2,3` implements GitHub issues end to end with headless `claude -p` (subscription auth): route each issue to a model, implement it with tests in its own worktree, re-verify locally, open a PR, watch CI, fix CI failures, and squash-merge on green.

**Architecture:** A bash entrypoint (`scripts/run_issue.sh`) orchestrates fixed phases — resolve, route, implement, ship, bookkeeping — and owns every irreversible step (push, PR, merge). Pure helpers live in `scripts/run_issue/lib.sh` so they can be unit-tested by sourcing. The agent only edits, tests, commits and writes `.run_issue/PR_BODY.md`, under an allowlist. Prompts are template files.

**Tech Stack:** bash 5, `git` worktrees, `gh` CLI (with `--jq`), `claude` CLI (`-p --output-format json`), `python3` (stdlib only: JSON and templating helpers), pytest for tests, shellcheck for lint.

**Spec:** `docs/superpowers/specs/2026-10-15-issue-runner-design.md`

## Global Constraints

- Every `claude` call runs with `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` removed from its environment (subscription login only).
- Model is passed as an alias: `opus`, `sonnet` or `fable`. `fable` only with `--allow-fable`; otherwise `[Fable]` → `opus` with note `Fable downgraded to Opus (no --allow-fable)`.
- Agent permissions: `--permission-mode acceptEdits` + `--allowedTools` from `scripts/run_issue/allowed_tools.txt`; never `--dangerously-skip-permissions`. The allowlist must not contain `git push`, `gh pr` or `gh api`.
- The prompt goes to `claude` on **stdin** (`claude -p` with no positional prompt), because `--allowedTools` is variadic.
- PR body headings, exactly: `## Summary`, `## Implementation details`, `## In plain words`, `## Verification`, `## Routing`. Footer: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Merge only after `gh pr checks --watch --fail-fast` exits 0 for a PR whose head SHA equals the pushed SHA and which reports ≥ 1 check. Merge with `gh pr merge --squash --delete-branch`.
- Defaults: `--repo wpawgasa/vlm-ocr-bench`, `--base main`, `--max-ci-fixes 2`; checks wait 300 s polled every 10 s (overridable by `RUN_ISSUE_CHECKS_WAIT_SECS` / `RUN_ISSUE_CHECKS_POLL_SECS` for tests).
- Exit codes: `0` all implementable issues merged (skips fine), `1` run halted, `2` usage/precondition error.
- Worktrees at `.claude/worktrees/issue-<n>`, branch `issue/<n>-<slug>`, logs at `.claude/worktrees/run-logs/<UTC timestamp>/`.
- No new runtime dependencies; no `jq` (use `gh --jq` and `python3`). `awk` is mawk on this host: POSIX regex only (`[ \t]`, no `\s`).
- The repository is public: prompts tell the agent never to put client data in PR bodies.
- Ship rule (memory): after implementation and verification, commit, push and open the PR without asking; **do not merge this plan's own PR** — the runner's auto-merge applies only to PRs the runner opens.

## Review Focus

1. **Issue text with shell/template metacharacters** (`$(…)`, backticks, `&`, `\`, `{{ISSUE_BODY}}`, Thai) — must reach the prompt verbatim and never execute or re-substitute. Pinned in Task 1 (`test_render_template_is_literal_and_single_pass`) and Task 5 (`test_issue_body_reaches_prompt_verbatim`).
2. **Stale CI status after a fix push** (PR head not yet updated, old checks visible) — must not merge on the old commit's checks. Pinned in Task 6 (`test_head_sha_mismatch_never_merges`).
3. **Parent issue bodies with CRLF line endings, look-alike refs (`#1170` vs `#117`) and already-ticked lines** — tick exactly one line, preserve everything else. Pinned in Task 1 (`test_tick_parent_line_*`).
4. **Titles with no ASCII letters (Thai-only, symbols)** — must still yield a valid branch name. Pinned in Task 1 (`test_branch_name_falls_back_for_non_ascii_title`).
5. **Rerun after a halted run** (worktree, branch and open PR already exist) — must reuse them, not re-implement or open a duplicate PR. Pinned in Task 6 (`test_rerun_reuses_worktree_and_pr`).

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/run_issue.sh` | Entrypoint: args, preconditions, per-issue phases, summary, exit code |
| `scripts/run_issue/lib.sh` | Side-effect-free helpers (parsing, routing table, templating, issue expansion) |
| `scripts/run_issue/implement.md` | Implementation prompt template |
| `scripts/run_issue/fix.md` | Fix prompt template (local or CI failure) |
| `scripts/run_issue/route.md` | Routing prompt template for untagged issues |
| `scripts/run_issue/allowed_tools.txt` | Agent allowlist, one entry per line |
| `tests/test_run_issue.py` | Unit + end-to-end tests |
| `tests/fixtures/run_issue/fake_tools.py` | Fake `gh` / `claude` / `uv` used by the e2e tests |
| `.gitignore` | `+ .run_issue/`, `+ .claude/worktrees/` |
| `.github/workflows/ci.yml` | `+ shellcheck` apt package |
| `README.md` | Short "Issue runner" usage section |

---

### Task 1: lib.sh pure helpers

**Files:**
- Create: `scripts/run_issue/lib.sh`
- Create: `tests/test_run_issue.py`
- Modify: `.gitignore` (append two lines)

**Interfaces:**
- Produces (bash functions, all print to stdout):
  - `log MSG...` → stderr (+ `$CURRENT_LOG` when set)
  - `parse_tier TITLE BODY` → `Sonnet|Opus|Human|Fable` or nothing
  - `strip_tier TITLE` → title without ` [Tier]`
  - `slugify TEXT` → `[a-z0-9-]`, ≤ 40 chars, may be empty
  - `branch_name N TITLE` → `issue/<n>-<slug or "issue">`
  - `resolve_model TIER ALLOW_FABLE(0|1)` → `<model>\t<note>`; model ∈ `sonnet|opus|fable|skip`
  - `commit_trailer MODEL` → `Co-Authored-By: …` line
  - `result_status TEXT` → `done` | `blocked: <reason>` | `missing`
  - `check_pr_body FILE` → missing-heading lines; return 1 if any
  - `tick_parent_line FILE N` → edits FILE in place; return 1 if no unchecked `(#N)` line
  - `render_template FILE KEY=VALUE...` → rendered text (single pass, `{{KEY}}`)
  - `parse_route_result TEXT` → `<Tier>\t<reason>` (fallback `Sonnet\trouting output unparseable; model-routing default`)
  - `json_field FILE KEY [DEFAULT]` → value or DEFAULT
  - `add_num A B` → A+B formatted `%.4f`
  - `load_tools FILE` → non-comment, non-blank lines

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_issue.py`:

```python
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
        ('{"tier": "Haiku", "reason": "cheap"}', "Sonnet\trouting output unparseable; model-routing default"),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_issue.py -q`
Expected: FAIL — every test errors because `scripts/run_issue/lib.sh` does not exist (bash: `No such file or directory`).

- [ ] **Step 3: Write `scripts/run_issue/lib.sh`**

```bash
# shellcheck shell=bash
# Function library for scripts/run_issue.sh. Sourced by the runner and by
# tests/test_run_issue.py; it only defines functions and constants.

export PYTHONUTF8=1

TIERS_RE='\[(Sonnet|Opus|Human|Fable)\]'
REQUIRED_PR_HEADINGS=(
  "## Summary"
  "## Implementation details"
  "## In plain words"
  "## Verification"
  "## Routing"
)

# log MSG... — timestamped line to stderr and, when set, to $CURRENT_LOG.
log() {
  local line
  line="[run_issue $(date -u +%H:%M:%S)] $*"
  printf '%s\n' "$line" >&2
  if [[ -n ${CURRENT_LOG:-} ]]; then
    printf '%s\n' "$line" >>"$CURRENT_LOG"
  fi
}

# parse_tier TITLE BODY — the tier tag from the title, else from a "**Tier:** [X]"
# body line; nothing when the issue is untagged.
parse_tier() {
  local title=$1 body=$2 line
  if [[ $title =~ $TIERS_RE ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  line=$(grep -m1 -E '^\*\*Tier:\*\*' <<<"$body" || true)
  if [[ $line =~ $TIERS_RE ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
}

# strip_tier TITLE — the title without its " [Tier]" tag.
strip_tier() {
  sed -E 's/ ?\[(Sonnet|Opus|Human|Fable)\]//' <<<"$1"
}

# slugify TEXT — lowercase ASCII words joined by "-", at most 40 characters.
slugify() {
  local s
  s=$(LC_ALL=C tr '[:upper:]' '[:lower:]' <<<"$1" |
    LC_ALL=C sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')
  s=${s:0:40}
  printf '%s\n' "${s%-}"
}

# branch_name N TITLE — "issue/<n>-<slug>"; the slug falls back to "issue" when
# the title has no ASCII letters or digits.
branch_name() {
  local slug
  slug=$(slugify "$(strip_tier "$2")")
  printf 'issue/%s-%s\n' "$1" "${slug:-issue}"
}

# resolve_model TIER ALLOW_FABLE — "<model>\t<note>". Model "skip" means a human
# must do the issue. Fable needs ALLOW_FABLE=1 (model-routing's confirmation gate).
resolve_model() {
  case $1 in
    Sonnet) printf 'sonnet\t\n' ;;
    Opus) printf 'opus\t\n' ;;
    Human) printf 'skip\tneeds a human\n' ;;
    Fable)
      if [[ $2 == 1 ]]; then
        printf 'fable\t\n'
      else
        printf 'opus\tFable downgraded to Opus (no --allow-fable)\n'
      fi
      ;;
    *) printf 'sonnet\tunknown tier %s; model-routing default\n' "$1" ;;
  esac
}

# commit_trailer MODEL — the Co-Authored-By trailer for commits made on MODEL.
commit_trailer() {
  case $1 in
    opus) echo "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" ;;
    fable) echo "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" ;;
    *) echo "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" ;;
  esac
}

# result_status TEXT — "done", "blocked: <reason>" or "missing", from the last
# "RESULT: " line of an agent's final message.
result_status() {
  local line
  line=$(grep -E '^RESULT: ' <<<"$1" | tail -n 1 || true)
  case $line in
    "RESULT: done"*) echo done ;;
    "RESULT: blocked"*) echo "blocked${line#RESULT: blocked}" ;;
    *) echo missing ;;
  esac
}

# check_pr_body FILE — print each required heading missing from FILE; return 1
# when the file or any heading is missing.
check_pr_body() {
  local file=$1 heading rc=0
  if [[ ! -f $file ]]; then
    echo "PR body file missing: $file"
    return 1
  fi
  for heading in "${REQUIRED_PR_HEADINGS[@]}"; do
    if ! grep -qE "^${heading}[[:space:]]*\$" "$file"; then
      echo "PR body missing heading: $heading"
      rc=1
    fi
  done
  return "$rc"
}

# tick_parent_line FILE N — change the first unchecked "- [ ] " line containing
# "(#N)" to "- [x] ". Returns 1 and leaves FILE untouched when none matches.
tick_parent_line() {
  local file=$1 n=$2 tmp
  tmp=$(mktemp)
  if awk -v ref="(#$n)" '
      !hit && index($0, ref) && /^[ \t]*- \[ \] / { sub(/- \[ \] /, "- [x] "); hit = 1 }
      { print }
      END { exit hit ? 0 : 1 }' "$file" >"$tmp"; then
    cat "$tmp" >"$file"
    rm -f "$tmp"
    return 0
  fi
  rm -f "$tmp"
  return 1
}

# render_template FILE KEY=VALUE... — FILE with each {{KEY}} replaced by VALUE in
# a single pass (values are never re-scanned); unknown placeholders stay as is.
render_template() {
  python3 - "$@" <<'PY'
import re
import sys

path, *pairs = sys.argv[1:]
values = dict(pair.split("=", 1) for pair in pairs)
with open(path, encoding="utf-8") as fh:
    text = fh.read()
sys.stdout.write(
    re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values.get(m.group(1), m.group(0)), text)
)
PY
}

# parse_route_result TEXT — "<Tier>\t<reason>" from the last JSON object in TEXT
# with a known tier; Sonnet (model-routing's default) when there is none.
parse_route_result() {
  python3 - "$1" <<'PY'
import json
import re
import sys

for match in reversed(list(re.finditer(r"\{[^{}]*\}", sys.argv[1]))):
    try:
        obj = json.loads(match.group(0))
    except ValueError:
        continue
    tier = str(obj.get("tier", "")).strip().capitalize()
    if tier in {"Sonnet", "Opus", "Human", "Fable"}:
        print(f"{tier}\t{' '.join(str(obj.get('reason', '')).split())}")
        break
else:
    print("Sonnet\trouting output unparseable; model-routing default")
PY
}

# json_field FILE KEY [DEFAULT] — top-level KEY of the JSON object in FILE, or
# DEFAULT when the file, the JSON or the key is missing.
json_field() {
  python3 - "$1" "$2" "${3:-}" <<'PY'
import json
import sys

path, key, default = sys.argv[1:4]
try:
    with open(path, encoding="utf-8") as fh:
        value = json.load(fh).get(key)
except (OSError, ValueError, AttributeError):
    value = None
print(default if value is None else value)
PY
}

# add_num A B — A + B to four decimals; empty operands count as 0.
add_num() {
  awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { printf "%.4f\n", a + b }'
}

# load_tools FILE — the allowlist entries in FILE (no comments or blank lines).
load_tools() {
  grep -vE '^[[:space:]]*(#|$)' "$1"
}
```

Append to `.gitignore`:

```
.run_issue/
.claude/worktrees/
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_issue.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check tests/test_run_issue.py && uv run ruff format tests/test_run_issue.py
git add scripts/run_issue/lib.sh tests/test_run_issue.py .gitignore
git commit -m "run_issue: lib.sh helpers (tier parsing, routing table, templating, PR body checks)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Fake tools and issue expansion

**Files:**
- Create: `tests/fixtures/run_issue/fake_tools.py`
- Modify: `scripts/run_issue/lib.sh` (append `issue_state`, `task_refs`, `expand_issues`)
- Modify: `tests/test_run_issue.py` (add `Fakes` helper + expansion tests)

**Interfaces:**
- Consumes: `call()` from Task 1 tests.
- Produces:
  - `issue_state REPO N` → `open` | `closed` (empty on error)
  - `task_refs` (stdin: issue body) → one `N` per line from `(#N)` checklist refs inside the `### Tasks` section
  - `expand_issues REPO N...` → lines `"<number> <state>"`, in order, sub-issues replacing their parent, duplicates dropped
  - Test helper class `Fakes(tmp_path)` with `.env`, `.gh(match, *responses, first=False)`, `.uv(match, *responses, first=False)`, `.claude(kind, *actions)`, `.calls(tool)`, `.reset()`; and `add_issue(fakes, n, title, body=..., state="open", parent=None, subs=None)`.

- [ ] **Step 1: Write the fake tools**

Create `tests/fixtures/run_issue/fake_tools.py`:

```python
"""Fake `gh`, `claude` and `uv` for tests/test_run_issue.py.

Called through wrappers on PATH as: fake_tools.py TOOL ARG...

Every call is appended to $FAKE_LOG as one JSON line. Replies come from the JSON
scenario at $FAKE_SCENARIO:

- "gh" / "uv": a list of rules {"match": regex, "responses": [{"stdout", "exit"}]}.
  The first rule whose regex matches the space-joined argv answers with its next
  response (the last one repeats). A stdout starting with "@SH:" is run with bash
  and its output is used instead. Unmatched calls print nothing and exit 0.
- "claude": {kind: [action, ...]}, kind taken from the prompt's
  "<!-- run_issue:KIND -->" marker. Action keys: "result" (final message, default
  "RESULT: done"), "commit" ("test" adds tests/test_fake_<k>.py, "code" adds
  fake_<k>.py), "pr_body" (write .run_issue/PR_BODY.md), "dirty" (leave an
  uncommitted file), "cost", "turns".

Per-rule call counters live in $FAKE_STATE.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PR_BODY = """## Summary
Fake change.

## Implementation details
Added a test.

## In plain words
It works now.

## Verification
pytest passed.

## Routing
sonnet
"""


def _next(key: str, items: list[dict]) -> dict:
    state_path = Path(os.environ["FAKE_STATE"])
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    index = state.get(key, 0)
    state[key] = index + 1
    state_path.write_text(json.dumps(state))
    return items[min(index, len(items) - 1)] if items else {}


def _log(entry: dict) -> int:
    log_path = Path(os.environ["FAKE_LOG"])
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return len(log_path.read_text(encoding="utf-8").splitlines())


def _rules(tool: str, argv: list[str], scenario: dict) -> int:
    entry: dict = {"tool": tool, "argv": argv, "cwd": os.getcwd()}
    if "--body-file" in argv:
        body_path = Path(argv[argv.index("--body-file") + 1])
        entry["body"] = body_path.read_text(encoding="utf-8")
    _log(entry)
    joined = " ".join(argv)
    for i, rule in enumerate(scenario.get(tool, [])):
        if re.search(rule["match"], joined):
            response = _next(f"{tool}:{i}", rule.get("responses", []))
            stdout = response.get("stdout", "")
            if stdout.startswith("@SH:"):
                stdout = subprocess.run(
                    ["bash", "-c", stdout[4:]], capture_output=True, text=True, check=False
                ).stdout
            sys.stdout.write(stdout)
            return int(response.get("exit", 0))
    return 0


def _claude(argv: list[str], scenario: dict) -> int:
    prompt = sys.stdin.read()
    marker = re.search(r"<!-- run_issue:([\w-]+) -->", prompt)
    kind = marker.group(1) if marker else "unknown"
    serial = _log(
        {
            "tool": "claude",
            "argv": argv,
            "cwd": os.getcwd(),
            "kind": kind,
            "prompt": prompt,
            "api_key_set": any(
                k in os.environ for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
            ),
        }
    )
    action = _next(f"claude:{kind}", scenario.get("claude", {}).get(kind, [{}]))
    cwd = Path.cwd()
    commit = action.get("commit")
    if commit:
        name = f"tests/test_fake_{serial}.py" if commit == "test" else f"fake_{serial}.py"
        (cwd / name).parent.mkdir(parents=True, exist_ok=True)
        (cwd / name).write_text("def test_fake():\n    assert True\n")
        subprocess.run(["git", "add", name], check=True)
        subprocess.run(["git", "commit", "-qm", f"fake commit {serial}"], check=True)
    if action.get("pr_body"):
        (cwd / ".run_issue").mkdir(exist_ok=True)
        (cwd / ".run_issue" / "PR_BODY.md").write_text(PR_BODY)
    if action.get("dirty"):
        (cwd / "dirty.txt").write_text("uncommitted\n")
    result = {
        "type": "result",
        "result": action.get("result", "RESULT: done"),
        "total_cost_usd": action.get("cost", 0.25),
        "num_turns": action.get("turns", 3),
    }
    print(json.dumps(result))
    return 0


def main() -> int:
    tool, argv = sys.argv[1], sys.argv[2:]
    scenario = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
    if tool == "claude":
        return _claude(argv, scenario)
    return _rules(tool, argv, scenario)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_run_issue.py` imports: `import json`, `import os`. Then append:

```python
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
            "responses": [r if isinstance(r, dict) else {"stdout": r} for r in responses]
            or [{}],
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_issue.py -q -k expand`
Expected: FAIL — `expand_issues: command not found`.

- [ ] **Step 4: Append to `scripts/run_issue/lib.sh`**

```bash
# issue_state REPO N — "open" or "closed" (empty when the issue cannot be read).
issue_state() {
  gh issue view "$2" -R "$1" --json state --jq '.state | ascii_downcase' 2>/dev/null
}

# task_refs — read an issue body on stdin; print N for each "(#N)" on a checklist
# line inside its "### Tasks" section.
task_refs() {
  awk '/^### Tasks/ { inside = 1; next } /^#/ { inside = 0 } inside && /^[ \t]*- \[[ xX]\] /' |
    grep -oE '\(#[0-9]+\)' | tr -d '(#)' || true
}

# expand_issues REPO N... — print "<number> <state>" for each issue to run, in
# order. An issue with sub-issues (GitHub sub-issue API, else the "(#N)" refs in
# its "### Tasks" section) is replaced by them. Duplicates are dropped.
expand_issues() {
  local repo=$1 n num line body
  shift
  local -A seen=()
  local -a rows
  for n in "$@"; do
    rows=()
    mapfile -t rows < <(gh api "repos/$repo/issues/$n/sub_issues" --paginate \
      --jq '.[] | "\(.number) \(.state)"' 2>/dev/null || true)
    if ((${#rows[@]} == 0)); then
      body=$(gh issue view "$n" -R "$repo" --json body --jq .body 2>/dev/null || true)
      for num in $(task_refs <<<"$body"); do
        rows+=("$num $(issue_state "$repo" "$num")")
      done
    fi
    if ((${#rows[@]} == 0)); then
      rows=("$n $(issue_state "$repo" "$n")")
    fi
    for line in "${rows[@]}"; do
      [[ -n $line ]] || continue
      num=${line%% *}
      [[ -z ${seen[$num]:-} ]] || continue
      seen[$num]=1
      printf '%s\n' "$line"
    done
  done
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_issue.py -q`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check tests && uv run ruff format tests/test_run_issue.py tests/fixtures/run_issue/fake_tools.py
git add scripts/run_issue/lib.sh tests/test_run_issue.py tests/fixtures/run_issue/fake_tools.py
git commit -m "run_issue: expand group issues into sub-issues; fake gh/claude/uv for tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Prompt templates and allowlist

**Files:**
- Create: `scripts/run_issue/implement.md`, `scripts/run_issue/fix.md`, `scripts/run_issue/route.md`, `scripts/run_issue/allowed_tools.txt`
- Modify: `tests/test_run_issue.py`

**Interfaces:**
- Produces template placeholders (exact names, used by Tasks 4–6):
  - `implement.md`: `{{REPO}} {{BASE}} {{ISSUE_NUMBER}} {{ISSUE_TITLE}} {{ROUTING}} {{TRAILER}} {{ISSUE_BODY}}`, marker `<!-- run_issue:implement -->`
  - `fix.md`: `{{ISSUE_NUMBER}} {{ISSUE_TITLE}} {{FAILURE_KIND}} {{TRAILER}} {{FAILURE_LOG}}`, marker `<!-- run_issue:fix -->`
  - `route.md`: `{{ISSUE_NUMBER}} {{ISSUE_TITLE}} {{ISSUE_BODY}}`, marker `<!-- run_issue:route -->`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_issue.py`:

```python
# --- templates and allowlist -------------------------------------------------

TEMPLATE_KEYS = {
    "implement.md": ["REPO", "BASE", "ISSUE_NUMBER", "ISSUE_TITLE", "ROUTING", "TRAILER", "ISSUE_BODY"],
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
        "uv run pytest -m \"not live\"",
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_issue.py -q -k "template or prompt or allowlist"`
Expected: FAIL — template files not found.

- [ ] **Step 3: Write the templates and allowlist**

`scripts/run_issue/implement.md`:

````markdown
<!-- run_issue:implement -->
You are implementing GitHub issue #{{ISSUE_NUMBER}} of {{REPO}} in this git worktree (a branch made from `origin/{{BASE}}`). You run unattended: nobody will answer questions. Work until the issue is done or you are genuinely blocked.

## Issue #{{ISSUE_NUMBER}}: {{ISSUE_TITLE}}

{{ISSUE_BODY}}

## Model routing

{{ROUTING}}

## How to work

1. Read `CLAUDE.md` if present, the files the issue references (OpenSpec `proposal.md`, `design.md`, `tasks.md`, specs) and the code around the change before writing anything.
2. Use the `superpowers:test-driven-development` skill. For every task in the issue, write a failing test under `tests/` first, run it and watch it fail, then implement until it passes. If the issue describes a bug, use `superpowers:systematic-debugging` to find the root cause before fixing it.
3. Tests must not need the network, GPUs or running model servers. Mark anything that does with `@pytest.mark.live`.
4. If the issue points at an OpenSpec `tasks.md`, tick this task's checkbox there (`- [ ]` → `- [x]`).
5. Use the `superpowers:verification-before-completion` skill. All of these must pass, and you must have seen them pass:
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run pytest -m "not live"`
6. Commit all of your changes; leave nothing uncommitted. The last commit message ends with:

   ```
   Closes #{{ISSUE_NUMBER}}

   {{TRAILER}}
   ```

7. Write `.run_issue/PR_BODY.md` (git-ignored; do not commit it) with exactly these headings, in this order:
   - `## Summary`: two or three sentences on what changed and why.
   - `## Implementation details`: the files you changed and what each change does, the design choices you made and why, and the tests you added (file, test names, what each one checks).
   - `## In plain words`: the same change explained to a non-programmer in a few short sentences: what was missing or wrong, and what works now. No jargon, no file names.
   - `## Verification`: the commands you ran and their results (for example "pytest: 412 passed").
   - `## Routing`: the model routing line above.

   The repository is public: never include client data (bank statement contents, account numbers, names).
8. Do not push, open pull requests or merge. The runner does that after you finish.

## Finish

End your final message with exactly one of these lines:

RESULT: done
RESULT: blocked: <one-line reason>

Use `blocked` only when the issue cannot be finished here: it needs a GPU, credentials, a human decision, or another unmerged issue.
````

`scripts/run_issue/fix.md`:

````markdown
<!-- run_issue:fix -->
You are continuing work on GitHub issue #{{ISSUE_NUMBER}} ({{ISSUE_TITLE}}) in this git worktree. The previous attempt failed {{FAILURE_KIND}}. You run unattended: nobody will answer questions.

## Failure output

~~~~text
{{FAILURE_LOG}}
~~~~

## What to do

1. Use the `superpowers:systematic-debugging` skill: find the root cause in the output above before changing code. Do not weaken, skip or delete tests to make them pass.
2. Fix it, keeping the issue's requirements and a test for every task in the issue.
3. Use `superpowers:verification-before-completion`: `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest -m "not live"` must pass.
4. Commit everything; leave nothing uncommitted. The commit message ends with `Refs #{{ISSUE_NUMBER}}`, a blank line, and `{{TRAILER}}`.
5. Make sure `.run_issue/PR_BODY.md` exists with the headings `## Summary`, `## Implementation details`, `## In plain words`, `## Verification` and `## Routing`, and that it describes the final change, including this fix.
6. Do not push, open pull requests or merge.

End your final message with exactly one line: `RESULT: done` or `RESULT: blocked: <one-line reason>`.
````

`scripts/run_issue/route.md`:

```markdown
<!-- run_issue:route -->
Decide which Claude model tier should implement GitHub issue #{{ISSUE_NUMBER}}. Do not implement anything and do not edit files.

Use the `model-routing` skill to classify the issue. You may read the repository to judge how hard the change is. Pick one tier:

- `Opus`: hard implementation (the skill's Opus criteria).
- `Sonnet`: routine, fully specified implementation.
- `Fable`: the issue is planning, spec or research work rather than implementation.
- `Human`: it needs a person, GPU hardware or credentials.

Do not ask for confirmation: the runner applies the Fable confirmation gate itself.

## Issue #{{ISSUE_NUMBER}}: {{ISSUE_TITLE}}

{{ISSUE_BODY}}

## Answer

Reply with one line of JSON and nothing after it, for example:

{"tier": "Sonnet", "reason": "<one sentence>"}
```

`scripts/run_issue/allowed_tools.txt`:

```
# Tools the implementing agent may use without a prompt (claude --allowedTools).
# Deliberately absent: git push, gh pr, gh api, curl/wget. The runner pushes,
# opens the PR and merges itself.
Read
Edit
Write
Glob
Grep
Skill
TodoWrite
Bash(uv run:*)
Bash(uv sync:*)
Bash(git add:*)
Bash(git rm:*)
Bash(git mv:*)
Bash(git commit:*)
Bash(git status:*)
Bash(git diff:*)
Bash(git log:*)
Bash(git show:*)
Bash(gh issue view:*)
Bash(openspec:*)
Bash(ls:*)
Bash(cat:*)
Bash(rg:*)
Bash(grep:*)
Bash(find:*)
Bash(head:*)
Bash(tail:*)
Bash(wc:*)
Bash(mkdir:*)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_issue.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format tests/test_run_issue.py
git add scripts/run_issue/*.md scripts/run_issue/allowed_tools.txt tests/test_run_issue.py
git commit -m "run_issue: implement/fix/route prompt templates and agent allowlist

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Runner skeleton — CLI, preconditions, routing, dry-run, summary

**Files:**
- Create: `scripts/run_issue.sh` (mode 755)
- Modify: `tests/test_run_issue.py`

**Interfaces:**
- Consumes: everything in `lib.sh`; templates from Task 3; `Fakes`/`add_issue` from Task 2.
- Produces (runner-internal, used by Tasks 5–6):
  - globals `ROOT REPO BASE LOG_DIR CURRENT_LOG IMPL_TOOLS[] ROUTE_TOOLS[] MAX_CI_FIXES CHECKS_WAIT_SECS CHECKS_POLL_SECS`, assoc arrays `OUT_MODEL OUT_OUTCOME OUT_PR OUT_COST OUT_TURNS`
  - `run_claude N KIND CWD MODEL PROMPT TOOL...` → sets `CLAUDE_RESULT`
  - `route_issue N TITLE BODY` → sets `ROUTE_TIER ROUTE_REASON`
  - `process_issue N` → 0 continue / 1 halt
  - Test fixture `sandbox` (a `Fakes` with `.work`, `.origin`) and `run_runner(sandbox, *args)`.

- [ ] **Step 1: Write the failing tests**

Add `import shutil` to the imports, then append:

```python
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
    [[], ["--issues", "abc"], ["--issues", "1,"], ["--issues", "1", "--max-ci-fixes", "x"], ["--bogus"], ["--issues"]],
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
    add_issue(sandbox, 116, "runpod-serving R1: group", subs=[(117, "open"), (118, "closed"), (119, "open")])
    add_issue(sandbox, 117, "runpod-serving 1.1 [Sonnet]: config loader")
    add_issue(sandbox, 118, "runpod-serving 1.2 [Sonnet]: configs")
    add_issue(sandbox, 119, "Untagged parity set")
    add_issue(sandbox, 120, "runpod-serving 5.4 [Human]: pin weights")
    sandbox.claude("route", {"result": 'ok\n{"tier": "Opus", "reason": "crosses modules"}'})

    result = run_runner(sandbox, "--issues", "116,120", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "-> sonnet (tier tag [Sonnet] on the issue)" in result.stdout
    assert "-> opus (model-routing skill: crosses modules)" in result.stdout
    summary = result.stdout[result.stdout.index("ISSUE"):]
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
    [([], "-> opus (tier tag [Fable] on the issue; Fable downgraded to Opus (no --allow-fable))"), (["--allow-fable"], "-> fable")],
)
def test_fable_gate(sandbox: Fakes, flags: list[str], expected: str) -> None:
    add_issue(sandbox, 5, "spec 0.1 [Fable]: draft design")
    result = run_runner(sandbox, "--issues", "5", "--dry-run", *flags)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_issue.py -q -k "usage or unauthenticated or dry_run or fable_gate"`
Expected: FAIL — `bash: .../scripts/run_issue.sh: No such file or directory` (exit 127).

- [ ] **Step 3: Write `scripts/run_issue.sh`**

Note: the script deliberately uses `set -uo pipefail` **without** `-e`: `set -e` is silently disabled inside functions called from `||` contexts, so every command that can fail is checked explicitly instead.

```bash
#!/usr/bin/env bash
# Issue runner: implement GitHub issues end to end with headless Claude Code.
#
#   scripts/run_issue.sh --issues 117,118,116 [--allow-fable] [--max-ci-fixes 2]
#                        [--dry-run] [--repo OWNER/NAME] [--base BRANCH]
#
# For each issue, in order (group issues expand into their sub-issues): pick the
# model from the issue's tier tag, or with the model-routing skill when untagged;
# implement it with tests in its own worktree via `claude -p` on the subscription
# login; re-verify locally; open a PR; watch CI and fix failures; squash-merge
# when CI passes. The run stops at the first issue that cannot be merged.
# Design: docs/superpowers/specs/2026-10-15-issue-runner-design.md
set -uo pipefail

RUN_ISSUE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/run_issue" && pwd)
# shellcheck source=scripts/run_issue/lib.sh
source "$RUN_ISSUE_DIR/lib.sh"

usage() {
  cat <<'EOF'
Usage: scripts/run_issue.sh --issues N[,N...] [--allow-fable] [--max-ci-fixes N]
                            [--dry-run] [--repo OWNER/NAME] [--base BRANCH]
EOF
}

die_usage() {
  printf 'run_issue: %s\n' "$1" >&2
  usage >&2
  exit 2
}

ISSUES="" ALLOW_FABLE=0 MAX_CI_FIXES=2 DRY_RUN=0
REPO="wpawgasa/vlm-ocr-bench" BASE="main"
CHECKS_WAIT_SECS=${RUN_ISSUE_CHECKS_WAIT_SECS:-300}
CHECKS_POLL_SECS=${RUN_ISSUE_CHECKS_POLL_SECS:-10}

while (($#)); do
  case $1 in
    --issues | --max-ci-fixes | --repo | --base)
      (($# >= 2)) || die_usage "$1 needs a value"
      case $1 in
        --issues) ISSUES=$2 ;;
        --max-ci-fixes) MAX_CI_FIXES=$2 ;;
        --repo) REPO=$2 ;;
        --base) BASE=$2 ;;
      esac
      shift 2
      ;;
    --allow-fable) ALLOW_FABLE=1 && shift ;;
    --dry-run) DRY_RUN=1 && shift ;;
    -h | --help) usage && exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
done

[[ $ISSUES =~ ^[0-9]+(,[0-9]+)*$ ]] || die_usage "--issues needs comma-separated issue numbers"
[[ $MAX_CI_FIXES =~ ^[0-9]+$ ]] || die_usage "--max-ci-fixes needs a non-negative integer"
for tool in claude gh git uv python3; do
  command -v "$tool" >/dev/null || die_usage "required tool not on PATH: $tool"
done
gh auth status >/dev/null 2>&1 || die_usage "gh is not authenticated (run: gh auth login)"
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die_usage "run this inside the repository"

mapfile -t IMPL_TOOLS < <(load_tools "$RUN_ISSUE_DIR/allowed_tools.txt")
ROUTE_TOOLS=(Read Glob Grep Skill)
LOG_DIR="$ROOT/.claude/worktrees/run-logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR" || die_usage "cannot create $LOG_DIR"
CURRENT_LOG="$LOG_DIR/run.log"
CLAUDE_CALLS=0 CLAUDE_RESULT="" ROUTE_TIER="" ROUTE_REASON=""
declare -A OUT_MODEL=() OUT_OUTCOME=() OUT_PR=() OUT_COST=() OUT_TURNS=()
ORDER=()

# run_claude N KIND CWD MODEL PROMPT TOOL... — one headless Claude Code call on
# the subscription login (API-key variables removed). The prompt goes on stdin
# because --allowedTools is variadic. Sets CLAUDE_RESULT to the final message and
# adds the call's cost and turns to issue N's totals.
run_claude() {
  local n=$1 kind=$2 cwd=$3 model=$4 prompt=$5 out
  shift 5
  CLAUDE_CALLS=$((CLAUDE_CALLS + 1))
  out="$LOG_DIR/issue-$n.$CLAUDE_CALLS.$kind.json"
  log "#$n: claude $kind (--model $model)"
  if ! (cd "$cwd" && printf '%s' "$prompt" |
    env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
      claude -p --model "$model" --permission-mode acceptEdits \
      --output-format json --allowedTools "$@") >"$out" 2>>"$CURRENT_LOG"; then
    log "#$n: claude $kind exited non-zero (see $out)"
  fi
  CLAUDE_RESULT=$(json_field "$out" result)
  OUT_COST[$n]=$(add_num "${OUT_COST[$n]:-0}" "$(json_field "$out" total_cost_usd 0)")
  OUT_TURNS[$n]=$((${OUT_TURNS[$n]:-0} + $(json_field "$out" num_turns 0)))
}

# route_issue N TITLE BODY — set ROUTE_TIER and ROUTE_REASON from the tier tag,
# or from a read-only model-routing call when the issue is untagged.
route_issue() {
  local n=$1 title=$2 body=$3 prompt
  ROUTE_TIER=$(parse_tier "$title" "$body")
  if [[ -n $ROUTE_TIER ]]; then
    ROUTE_REASON="tier tag [$ROUTE_TIER] on the issue"
    return 0
  fi
  prompt=$(render_template "$RUN_ISSUE_DIR/route.md" \
    "ISSUE_NUMBER=$n" "ISSUE_TITLE=$title" "ISSUE_BODY=$body")
  run_claude "$n" route "$ROOT" sonnet "$prompt" "${ROUTE_TOOLS[@]}"
  IFS=$'\t' read -r ROUTE_TIER ROUTE_REASON < <(parse_route_result "$CLAUDE_RESULT")
  ROUTE_REASON="model-routing skill: $ROUTE_REASON"
}

# process_issue N — route, implement and ship one open issue. Returns 1 when the
# run must halt.
process_issue() {
  local n=$1 title body model note reason
  CURRENT_LOG="$LOG_DIR/issue-$n.log"
  OUT_OUTCOME[$n]=error
  if ! title=$(gh issue view "$n" -R "$REPO" --json title --jq .title) ||
    ! body=$(gh issue view "$n" -R "$REPO" --json body --jq .body); then
    log "#$n: cannot read the issue"
    return 1
  fi
  route_issue "$n" "$title" "$body"
  IFS=$'\t' read -r model note < <(resolve_model "$ROUTE_TIER" "$ALLOW_FABLE")
  reason=$ROUTE_REASON
  [[ -z $note ]] || reason+="; $note"
  OUT_MODEL[$n]=$model
  if [[ $model == skip ]]; then
    OUT_OUTCOME[$n]=skipped-human
    log "#$n: $reason; skipped"
    return 0
  fi
  if ((DRY_RUN)); then
    OUT_OUTCOME[$n]=planned
    printf '#%s %s\n    -> %s (%s)\n' "$n" "$title" "$model" "$reason"
    return 0
  fi
  log "#$n: implementation phase not available yet"
  return 1
}

print_summary() {
  local n fmt='%-7s %-13s %-15s %-9s %-6s %s\n'
  printf '\n'
  # shellcheck disable=SC2059
  printf "$fmt" ISSUE MODEL OUTCOME COST_USD TURNS PR
  for n in "${ORDER[@]}"; do
    # shellcheck disable=SC2059
    printf "$fmt" "#$n" "${OUT_MODEL[$n]:--}" "${OUT_OUTCOME[$n]:-error}" \
      "${OUT_COST[$n]:-0}" "${OUT_TURNS[$n]:-0}" "${OUT_PR[$n]:--}"
  done
  printf 'Logs: %s\n' "$LOG_DIR"
}

IFS=, read -ra REQUESTED <<<"$ISSUES"
mapfile -t ROWS < <(expand_issues "$REPO" "${REQUESTED[@]}")
HALTED=0
for row in "${ROWS[@]}"; do
  n=${row%% *}
  state=${row#* }
  ORDER+=("$n")
  case $state in
    open) process_issue "$n" || HALTED=1 ;;
    closed)
      OUT_OUTCOME[$n]=skipped-closed
      log "#$n: closed; skipped"
      ;;
    *)
      OUT_OUTCOME[$n]=error
      log "#$n: cannot read the issue state"
      HALTED=1
      ;;
  esac
  ((HALTED == 0)) || break
done
print_summary
exit "$HALTED"
```

Then: `chmod +x scripts/run_issue.sh`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_issue.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format tests/test_run_issue.py && uv run ruff check tests
git add scripts/run_issue.sh tests/test_run_issue.py
git commit -m "run_issue: CLI, preconditions, routing (tags, model-routing, Fable gate), dry-run, summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Implement phase — worktree, agent call, post-checks, fix rounds, escalation

**Files:**
- Modify: `scripts/run_issue.sh`
- Modify: `tests/test_run_issue.py`

**Interfaces:**
- Consumes: `run_claude`, `render_template`, `branch_name`, `commit_trailer`, `result_status`, `check_pr_body`.
- Produces:
  - `implement_issue N TITLE BODY MODEL REASON` → 0 ready / 1 halt; sets globals `ISSUE_WT`, `ISSUE_BRANCH`, `ISSUE_MODEL` (model after any escalation)
  - `post_checks N STATUS` → prints the failure, returns 1; returns 0 when ready
  - On success `process_issue` sets outcome `implemented` (Task 6 replaces this with shipping)
  - Test helper `ship_ok(sandbox, n)` scripting a green PR for issue n.

- [ ] **Step 1: Write the failing tests**

Append:

```python
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
        ({"commit": "test", "pr_body": True, "result": "I think it is fine."}, "did not end with 'RESULT: done'"),
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


@pytest.mark.parametrize(("tier", "fix_models"), [("Sonnet", ["sonnet", "opus"]), ("Opus", ["opus"])])
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
```

Only the first fix prompt is asserted: in the `dirty` case `dirty.txt` stays uncommitted, so later rounds keep failing and the run halts, which is expected.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_issue.py -q -k "implement or verbatim or blocked or post_check or escalat"`
Expected: FAIL — runner exits 1 with "implementation phase not available yet"; no `implement` claude calls.

- [ ] **Step 3: Add the implement phase to `scripts/run_issue.sh`**

Insert these functions after `route_issue`:

```bash
# post_checks N STATUS — print why the agent's work on issue N is not ready and
# return 1; return 0 when it finished, committed everything, changed tests, wrote
# the PR body, and lint plus tests pass in the worktree.
post_checks() {
  local status=$2 out dirty
  if [[ $status != done ]]; then
    echo "The agent did not end with 'RESULT: done' (got: $status)."
    return 1
  fi
  if [[ -z $(git -C "$ISSUE_WT" rev-list "origin/$BASE..HEAD") ]]; then
    echo "No commits on the branch beyond origin/$BASE."
    return 1
  fi
  dirty=$(git -C "$ISSUE_WT" status --porcelain -- . ':!.run_issue')
  if [[ -n $dirty ]]; then
    printf 'Uncommitted changes left in the worktree:\n%s\n' "$dirty"
    return 1
  fi
  check_pr_body "$ISSUE_WT/.run_issue/PR_BODY.md" || return 1
  if ! git -C "$ISSUE_WT" diff --name-only "origin/$BASE..HEAD" | grep -q '^tests/'; then
    echo "No files under tests/ changed: every task in the issue needs a test."
    return 1
  fi
  if ! out=$(cd "$ISSUE_WT" && uv run ruff check . 2>&1 &&
    uv run ruff format --check . 2>&1 && uv run pytest -m "not live" -q 2>&1); then
    printf 'Local verification failed:\n%s\n' "$(tail -n 200 <<<"$out")"
    return 1
  fi
}

# implement_issue N TITLE BODY MODEL REASON — create or reuse the issue's
# worktree, run the implementing agent and check its work: one fix round on
# MODEL, then one on Opus when MODEL is Sonnet. Sets ISSUE_WT, ISSUE_BRANCH and
# ISSUE_MODEL. Returns 1 when the run must halt.
implement_issue() {
  local n=$1 title=$2 body=$3 model=$4 reason=$5 prompt status failure m reused=0
  ISSUE_BRANCH=$(branch_name "$n" "$title")
  ISSUE_WT="$ROOT/.claude/worktrees/issue-$n"
  ISSUE_MODEL=$model
  git -C "$ROOT" fetch -q origin >>"$CURRENT_LOG" 2>&1 || {
    log "#$n: git fetch failed"
    return 1
  }
  if [[ -d $ISSUE_WT ]]; then
    reused=1
  elif git -C "$ROOT" show-ref --verify --quiet "refs/heads/$ISSUE_BRANCH"; then
    reused=1
    git -C "$ROOT" worktree add -q "$ISSUE_WT" "$ISSUE_BRANCH" >>"$CURRENT_LOG" 2>&1 || return 1
  else
    git -C "$ROOT" worktree add -q -b "$ISSUE_BRANCH" "$ISSUE_WT" "origin/$BASE" \
      >>"$CURRENT_LOG" 2>&1 || {
      log "#$n: cannot create worktree $ISSUE_WT"
      return 1
    }
  fi
  (cd "$ISSUE_WT" && uv sync --all-extras --frozen) >>"$CURRENT_LOG" 2>&1 || {
    log "#$n: uv sync failed"
    return 1
  }
  if ((reused)) && post_checks "$n" done >/dev/null; then
    log "#$n: reusing $ISSUE_BRANCH, which already passes the checks"
    return 0
  fi

  prompt=$(render_template "$RUN_ISSUE_DIR/implement.md" "REPO=$REPO" "BASE=$BASE" \
    "ISSUE_NUMBER=$n" "ISSUE_TITLE=$title" "ROUTING=$model ($reason)" \
    "TRAILER=$(commit_trailer "$model")" "ISSUE_BODY=$body")
  run_claude "$n" implement "$ISSUE_WT" "$model" "$prompt" "${IMPL_TOOLS[@]}"

  local -a fix_models=("$model")
  [[ $model != sonnet ]] || fix_models+=(opus)
  for m in "${fix_models[@]}" ""; do
    status=$(result_status "$CLAUDE_RESULT")
    if [[ $status == blocked* ]]; then
      OUT_OUTCOME[$n]=blocked
      log "#$n: agent reported $status"
      return 1
    fi
    if failure=$(post_checks "$n" "$status"); then
      return 0
    fi
    [[ -n $m ]] || break
    log "#$n: checks failed ($(head -n 1 <<<"$failure")); fix round on $m"
    [[ $m == "$ISSUE_MODEL" ]] || OUT_MODEL[$n]+=">$m"
    ISSUE_MODEL=$m
    prompt=$(render_template "$RUN_ISSUE_DIR/fix.md" "ISSUE_NUMBER=$n" \
      "ISSUE_TITLE=$title" "FAILURE_KIND=local verification" \
      "TRAILER=$(commit_trailer "$m")" "FAILURE_LOG=$failure")
    run_claude "$n" fix "$ISSUE_WT" "$m" "$prompt" "${IMPL_TOOLS[@]}"
  done
  OUT_OUTCOME[$n]=blocked
  log "#$n: still failing after the fix rounds; worktree kept at $ISSUE_WT"
  return 1
}
```

In `process_issue`, replace:

```bash
  log "#$n: implementation phase not available yet"
  return 1
}
```

with:

```bash
  implement_issue "$n" "$title" "$body" "$model" "$reason" || return 1
  OUT_OUTCOME[$n]=implemented
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_issue.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format tests/test_run_issue.py && uv run ruff check tests
git add scripts/run_issue.sh tests/test_run_issue.py
git commit -m "run_issue: implement phase (worktree, agent call, post-checks, fix rounds, Opus escalation)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Ship phase — PR, CI loop, merge, bookkeeping

**Files:**
- Modify: `scripts/run_issue.sh`
- Modify: `tests/test_run_issue.py`

**Interfaces:**
- Consumes: `ISSUE_WT ISSUE_BRANCH ISSUE_MODEL`, `post_checks`, `run_claude`, `tick_parent_line`, `strip_tier`; `ship_ok` from Task 5 tests.
- Produces: `ship_issue N TITLE` → 0 merged / 1 halt; helpers `write_pr_body FILE`, `wait_for_checks PR SHA`, `failed_ci_log SHA`, `stop_pr N PR WHY`, `after_merge N PR`. Outcomes `merged`, `ci-failed`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
# --- end to end: ship phase --------------------------------------------------


def origin_has_branch(sandbox: Fakes, n: int) -> bool:
    refs = subprocess.run(
        ["git", f"--git-dir={sandbox.origin}", "for-each-ref", "--format=%(refname)", f"refs/heads/issue/{n}-*"],
        capture_output=True, text=True, check=True,
    ).stdout
    return bool(refs.strip())


def test_happy_path_expands_group_and_merges(sandbox: Fakes) -> None:
    parent_body = "### Tasks\n- [ ] 1.1 [Sonnet] loader (#117)\n- [ ] 1.2 [Sonnet] configs (#118)\n"
    add_issue(sandbox, 116, "runpod-serving R1: group", body=parent_body, subs=[(117, "open"), (118, "closed")])
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
    assert merge["argv"][2:] == ["-R", REPO, "--squash", "--delete-branch"]
    (edit,) = sandbox.gh_calls("issue edit 116")
    assert "- [x] 1.1 [Sonnet] loader (#117)" in edit["body"]
    assert "- [ ] 1.2 [Sonnet] configs (#118)" in edit["body"]
    assert sandbox.gh_calls("issue comment 117")
    assert not (sandbox.work / ".claude" / "worktrees" / "issue-117").exists()
    summary = result.stdout[result.stdout.index("ISSUE"):]
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
    sandbox.gh(rf"^pr list -R {REPO} --head issue/117-", f"https://github.com/{REPO}/pull/117\n", first=True)

    result = run_runner(sandbox, "--issues", "117")

    assert result.returncode == 0, result.stderr
    assert not claude_calls(sandbox)
    assert not sandbox.gh_calls("pr create")
    assert sandbox.gh_calls("pr edit")
    assert len(sandbox.gh_calls("pr merge")) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_issue.py -q -k "happy or ci_ or no_checks or head_sha or rerun"`
Expected: FAIL — no `pr create`/`pr merge` calls; summary says `implemented`.

- [ ] **Step 3: Add the ship phase to `scripts/run_issue.sh`**

Insert after `implement_issue`:

```bash
# write_pr_body FILE — the agent's PR body plus the Claude Code footer.
write_pr_body() {
  local footer='🤖 Generated with [Claude Code](https://claude.com/claude-code)'
  {
    cat "$ISSUE_WT/.run_issue/PR_BODY.md"
    grep -qF "$footer" "$ISSUE_WT/.run_issue/PR_BODY.md" || printf '\n\n%s\n' "$footer"
  } >"$1"
}

# wait_for_checks PR SHA — wait until the PR's head is SHA and at least one check
# is reported for it. Returns 1 after CHECKS_WAIT_SECS without that.
wait_for_checks() {
  local pr=$1 sha=$2 waited=0 head count
  while ((waited <= CHECKS_WAIT_SECS)); do
    head=$(gh pr view "$pr" -R "$REPO" --json headRefOid --jq .headRefOid 2>/dev/null)
    if [[ $head == "$sha" ]]; then
      count=$(gh pr checks "$pr" -R "$REPO" --json name --jq length 2>/dev/null)
      [[ $count =~ ^[0-9]+$ ]] || count=0
      ((count == 0)) || return 0
    fi
    sleep "$CHECKS_POLL_SECS"
    waited=$((waited + CHECKS_POLL_SECS))
    ((CHECKS_POLL_SECS > 0)) || break
  done
  return 1
}

# failed_ci_log SHA — the last 200 lines of the newest failed run's log for SHA.
failed_ci_log() {
  local id
  id=$(gh run list -R "$REPO" --commit "$1" --status failure --limit 1 \
    --json databaseId --jq '.[0].databaseId // empty' 2>/dev/null)
  if [[ -n $id ]]; then
    gh run view "$id" -R "$REPO" --log-failed 2>&1 | tail -n 200
  else
    echo "CI failed for $1, but no failed run log was found."
  fi
}

# stop_pr N PR WHY — record a CI failure, leave the PR open and say why on it.
stop_pr() {
  local n=$1 pr=$2 why=$3
  OUT_OUTCOME[$n]=ci-failed
  log "#$n: $why; leaving $pr open"
  gh pr comment "$pr" -R "$REPO" --body "scripts/run_issue.sh stopped: $why. \
This PR was not merged; the worktree .claude/worktrees/issue-$n is kept for inspection." \
    >>"$CURRENT_LOG" 2>&1 || true
}

# after_merge N PR — tick issue N in its parent's task list, comment the PR link
# on the issue, and remove the worktree and local branch.
after_merge() {
  local n=$1 pr=$2 parent body_file
  parent=$(gh api "repos/$REPO/issues/$n/parent" --jq .number 2>/dev/null)
  if [[ $parent =~ ^[0-9]+$ ]]; then
    body_file="$LOG_DIR/issue-$parent.body.md"
    if gh issue view "$parent" -R "$REPO" --json body --jq .body >"$body_file" &&
      tick_parent_line "$body_file" "$n"; then
      gh issue edit "$parent" -R "$REPO" --body-file "$body_file" >>"$CURRENT_LOG" 2>&1 ||
        log "#$n: could not tick the line in parent #$parent"
    fi
  fi
  gh issue comment "$n" -R "$REPO" --body "Merged in $pr by scripts/run_issue.sh." \
    >>"$CURRENT_LOG" 2>&1 || log "#$n: could not comment on the issue"
  git -C "$ROOT" worktree remove --force "$ISSUE_WT" >>"$CURRENT_LOG" 2>&1 || true
  git -C "$ROOT" branch -D "$ISSUE_BRANCH" >>"$CURRENT_LOG" 2>&1 || true
}

# ship_issue N TITLE — push, open or reuse the PR, wait for CI, run up to
# MAX_CI_FIXES fix rounds, and squash-merge when CI passes. Returns 1 when the
# run must halt.
ship_issue() {
  local n=$1 title=$2 pr sha body_file fixes=0 failure="" kind prompt status
  body_file="$LOG_DIR/issue-$n.pr-body.md"
  write_pr_body "$body_file"
  git -C "$ISSUE_WT" push -q -u origin "$ISSUE_BRANCH" >>"$CURRENT_LOG" 2>&1 || {
    log "#$n: git push failed"
    return 1
  }
  pr=$(gh pr list -R "$REPO" --head "$ISSUE_BRANCH" --state open --json url \
    --jq '.[0].url // empty' 2>/dev/null)
  if [[ -n $pr ]]; then
    log "#$n: reusing $pr"
    gh pr edit "$pr" -R "$REPO" --body-file "$body_file" >>"$CURRENT_LOG" 2>&1
  elif ! pr=$(gh pr create -R "$REPO" --base "$BASE" --head "$ISSUE_BRANCH" \
    --title "$(strip_tier "$title") (#$n)" --body-file "$body_file") || [[ -z $pr ]]; then
    log "#$n: gh pr create failed"
    return 1
  fi
  pr=$(tail -n 1 <<<"$pr")
  OUT_PR[$n]=$pr

  while true; do
    if [[ -z $failure ]]; then
      sha=$(git -C "$ISSUE_WT" rev-parse HEAD)
      if ! wait_for_checks "$pr" "$sha"; then
        stop_pr "$n" "$pr" "no CI checks reported for $sha within ${CHECKS_WAIT_SECS}s"
        return 1
      fi
      if gh pr checks "$pr" -R "$REPO" --watch --fail-fast >>"$CURRENT_LOG" 2>&1; then
        if ! gh pr merge "$pr" -R "$REPO" --squash --delete-branch >>"$CURRENT_LOG" 2>&1; then
          stop_pr "$n" "$pr" "CI passed but gh pr merge failed"
          return 1
        fi
        OUT_OUTCOME[$n]=merged
        log "#$n: merged $pr"
        after_merge "$n" "$pr"
        return 0
      fi
      failure=$(failed_ci_log "$sha")
      kind="on CI (GitHub Actions)"
    fi
    if ((fixes >= MAX_CI_FIXES)); then
      stop_pr "$n" "$pr" "CI still failing after $fixes fix round(s)"
      return 1
    fi
    fixes=$((fixes + 1))
    log "#$n: failed $kind; fix round $fixes/$MAX_CI_FIXES on $ISSUE_MODEL"
    prompt=$(render_template "$RUN_ISSUE_DIR/fix.md" "ISSUE_NUMBER=$n" \
      "ISSUE_TITLE=$title" "FAILURE_KIND=$kind" \
      "TRAILER=$(commit_trailer "$ISSUE_MODEL")" "FAILURE_LOG=$failure")
    run_claude "$n" ci-fix "$ISSUE_WT" "$ISSUE_MODEL" "$prompt" "${IMPL_TOOLS[@]}"
    status=$(result_status "$CLAUDE_RESULT")
    if [[ $status == blocked* ]]; then
      stop_pr "$n" "$pr" "the agent reported $status"
      return 1
    fi
    if failure=$(post_checks "$n" "$status"); then
      write_pr_body "$body_file"
      git -C "$ISSUE_WT" push -q origin "$ISSUE_BRANCH" >>"$CURRENT_LOG" 2>&1 || {
        stop_pr "$n" "$pr" "git push of the fix failed"
        return 1
      }
      gh pr edit "$pr" -R "$REPO" --body-file "$body_file" >>"$CURRENT_LOG" 2>&1
    else
      kind="local verification after a CI fix"
    fi
  done
}
```

In `process_issue`, replace:

```bash
  implement_issue "$n" "$title" "$body" "$model" "$reason" || return 1
  OUT_OUTCOME[$n]=implemented
}
```

with:

```bash
  implement_issue "$n" "$title" "$body" "$model" "$reason" || return 1
  ship_issue "$n" "$title"
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_issue.py -q`
Expected: all PASS (including Task 5's tests, which already script a green PR via `ship_ok`).

- [ ] **Step 5: Commit**

```bash
uv run ruff format tests/test_run_issue.py && uv run ruff check tests
git add scripts/run_issue.sh tests/test_run_issue.py
git commit -m "run_issue: ship phase (PR, CI watch with head-SHA guard, CI fix rounds, squash-merge, parent tick)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: shellcheck, docs, full verification, PR

**Files:**
- Modify: `tests/test_run_issue.py`
- Modify: `.github/workflows/ci.yml` (apt line)
- Modify: `README.md` (append a section)

- [ ] **Step 1: Write the shellcheck test**

Append:

```python
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
```

- [ ] **Step 2: Install shellcheck locally and run it**

Run: `sudo apt-get install -y shellcheck || uv tool install shellcheck-py`
Then: `uv run pytest tests/test_run_issue.py -q -k shellcheck`
Expected: PASS. If shellcheck reports findings, fix them in the scripts (do not add blanket disables; a targeted `# shellcheck disable=SCxxxx` with a reason is acceptable only where the code is intentionally that way, as for the `printf "$fmt"` lines).

- [ ] **Step 3: Add shellcheck to CI**

In `.github/workflows/ci.yml`, change:

```yaml
      # poppler: pdffonts/pdftoppm/pdftotext for the bank-statement ingestion tests
      - name: System packages
        run: apt-get update && apt-get install -y --no-install-recommends poppler-utils
```

to:

```yaml
      # poppler: pdffonts/pdftoppm/pdftotext for the bank-statement ingestion tests
      # git: issue-runner end-to-end tests; shellcheck: lints scripts/run_issue*
      - name: System packages
        run: apt-get update && apt-get install -y --no-install-recommends poppler-utils git shellcheck
```

- [ ] **Step 4: Document usage in `README.md`**

Append:

```markdown
## Issue runner

`scripts/run_issue.sh` implements GitHub issues end to end with headless Claude Code on your subscription login:

    scripts/run_issue.sh --issues 117,118          # sub-issues, in order
    scripts/run_issue.sh --issues 116 --dry-run    # expand a group issue and show the model plan

Per issue it picks the model from the `[Sonnet]`/`[Opus]` tag (or the `model-routing` skill when untagged; `[Human]` is skipped; `[Fable]` needs `--allow-fable`), implements it with tests in `.claude/worktrees/issue-<n>`, opens a PR, fixes CI failures (`--max-ci-fixes`, default 2) and squash-merges when CI passes. It stops at the first issue it cannot merge; rerun the same command to resume. Logs: `.claude/worktrees/run-logs/`. Design: `docs/superpowers/specs/2026-10-15-issue-runner-design.md`.
```

- [ ] **Step 5: Full local verification**

Run each and read the output:
```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not live" -q
bash scripts/run_issue.sh --issues 116 --dry-run
```
Expected: ruff clean; full suite passes; the real dry-run against GitHub lists #117–#119 (expanded from #116) with their tier-tag models and makes no changes (it creates only a log directory).

- [ ] **Step 6: Commit, push, open the PR, watch CI**

```bash
git add tests/test_run_issue.py .github/workflows/ci.yml README.md scripts/
git commit -m "run_issue: shellcheck in CI and tests; README usage

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin issue-runner
gh pr create -R wpawgasa/vlm-ocr-bench --base main --head issue-runner \
  --title "Issue runner: implement GitHub issues end to end with headless Claude Code" \
  --body-file <body written with Summary / Implementation details / In plain words / Verification sections + footer>
gh pr checks --watch
```
Expected: CI green. Do **not** merge this PR (the user merges it).

---

## Self-Review Notes

- Spec coverage: CLI + preconditions + subscription auth (T4); expansion incl. fallback, closed, dedupe (T2); routing table, routing call, Fable gate (T1, T4); worktree/reuse, uv sync, agent call, allowlist, prompt contents, post-checks, fix + Opus escalation, blocked halt (T3, T5); push, PR create/reuse, checks wait, head-SHA guard, CI fix loop, merge, comment on halt (T6); parent tick, issue comment, worktree cleanup (T6); logs + summary + exit codes (T4, T6); tests incl. shellcheck (T1–T7).
- Deviation from spec, deliberate: JSON fields are read with `python3` (stdlib) instead of `gh --jq` where the input is a local file (claude output) — `gh --jq` cannot read local files and the spec forbids a `jq` dependency. The prompt is passed on stdin rather than as an argument (variadic `--allowedTools`).
- `post_checks` also refuses uncommitted changes (not in the spec's list, needed so the pushed branch equals what was verified).
````
