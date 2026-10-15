# Issue runner (`scripts/run_issue.sh`) — design

**Date:** 2026-10-15
**Status:** approved in brainstorming, pending spec review

## Goal

`scripts/run_issue.sh --issues 1,2,3` takes GitHub issue numbers from
`wpawgasa/vlm-ocr-bench` and, per issue, has a headless Claude Code agent
(`claude -p`, authenticated by the user's existing subscription) implement the
task with tests, verifies it locally, opens a PR, watches CI, and squash-merges
the PR when CI passes.

Success: a run over a list of tagged sub-issues ends with one merged PR per
implementable issue, each PR body explaining the change in detail and in plain
words, with no human input, and never merges a PR whose CI did not pass.

## Decisions

| # | Decision |
|---|---|
| 1 | One branch + PR per issue, issues processed **sequentially**; each PR merges before the next issue starts from fresh `origin/main`. |
| 2 | Group issues are expanded into their sub-issues. |
| 3 | Headless agent runs with an **allowlist** (`--permission-mode acceptEdits` + `--allowedTools`), never `--dangerously-skip-permissions`. |
| 4 | **Deterministic shell orchestration**: the shell owns routing, push, PR creation, CI watching and merging; the agent owns code, tests, commits and the PR body. |
| 5 | Auto-merge is explicitly requested by the user for this runner. It overrides the "never merge" memory rule **only inside the runner**. |
| 6 | Fable is never selected without `--allow-fable` (model-routing's confirmation gate; the flag is the user's explicit yes for that run). |
| 7 | Model is passed as the alias `opus` / `sonnet` / `fable` so the runner tracks the current model versions (model-routing's table names stale versions). |

## CLI

```
scripts/run_issue.sh --issues 117,118,116 [--allow-fable] [--max-ci-fixes 2]
                     [--dry-run] [--repo wpawgasa/vlm-ocr-bench] [--base main]
```

- `--issues` (required): comma-separated issue numbers.
- `--allow-fable`: permit the Fable tier (otherwise downgraded to Opus, noted in the summary and PR body).
- `--max-ci-fixes N` (default 2): CI fix rounds per PR.
- `--dry-run`: resolve and route issues, print the plan (issue, title, model, reason); no worktree, no agent implementation call, no push, no PR. Routing of untagged issues still calls the routing agent (read-only).
- `--repo` (default `wpawgasa/vlm-ocr-bench`), `--base` (default `main`).

**Preconditions** (exit 2 on failure): `claude`, `gh`, `git`, `uv` on `PATH`; `gh auth status` succeeds; run from inside the repo.

**Subscription auth:** every `claude` invocation runs with `ANTHROPIC_API_KEY` (and `ANTHROPIC_AUTH_TOKEN`) unset, so the CLI uses the logged-in subscription and never falls back to API billing.

## Phase 1 — Resolve issues

For each number in order:

1. `gh api repos/{repo}/issues/{n}/sub_issues` → if non-empty, replace `n` by its sub-issues in API order.
2. Else, if the body has a `### Tasks` list with `(#NNN)` refs, treat those as sub-issues (fallback).
3. Else `n` is a leaf.

Closed issues are skipped with a notice. Duplicates are dropped, keeping first occurrence. Expansion is one level deep.

## Phase 2 — Route

Tier source, first match wins: a `[Sonnet]|[Opus]|[Human]|[Fable]` tag in the title, then a `**Tier:** [X]` line in the body.

| Tier | Action |
|---|---|
| Sonnet | `--model sonnet` |
| Opus | `--model opus` |
| Human | skip; outcome `skipped-human` |
| Fable | `--model fable` if `--allow-fable`, else `--model opus` with note "Fable downgraded to Opus (no --allow-fable)" |
| none | routing call (below) |

**Routing call** (untagged issues): `claude -p` with `--model sonnet`, read-only tools (`Read,Glob,Grep,Skill`), prompt `scripts/run_issue/route.md` filled with the issue title and body, instructing it to apply the `model-routing` skill and print only `{"tier":"Opus|Sonnet|Human|Fable","reason":"…"}`. The runner extracts the last JSON object from the result. Unparseable result → Sonnet, reason "routing output unparseable; model-routing default". The Fable gate applies to the routed tier.

## Phase 3 — Implement

1. `git fetch origin`; `git worktree add .claude/worktrees/issue-<n> -b issue/<n>-<slug> origin/<base>` (reuse branch/worktree if they exist — rerun/resume case). `slug` = title without tier tag, lowercased, non-alphanumerics → `-`, max 40 chars.
2. `uv sync --all-extras --frozen` in the worktree.
3. Agent call, cwd = worktree:

   ```
   claude -p "$(render implement.md)" --model <m> --permission-mode acceptEdits \
     --allowedTools "<allowed_tools.txt joined by ,>" --output-format json
   ```

   **Allowlist** (`scripts/run_issue/allowed_tools.txt`): `Read`, `Edit`, `Write`, `Glob`, `Grep`, `Skill`, `TodoWrite`, `Bash(uv run:*)`, `Bash(uv sync:*)`, `Bash(git add:*)`, `Bash(git commit:*)`, `Bash(git status:*)`, `Bash(git diff:*)`, `Bash(git log:*)`, `Bash(gh issue view:*)`, `Bash(openspec:*)`, `Bash(ls:*)`, `Bash(cat:*)`, `Bash(rg:*)`, `Bash(find:*)`, `Bash(head:*)`, `Bash(tail:*)`, `Bash(mkdir:*)`. Deliberately excluded: `git push`, `gh pr`, `gh api`, network tools.

   **Prompt** (`scripts/run_issue/implement.md`) contains the issue number, title, body, routing reason, and instructs the agent to:
   1. Use `superpowers:test-driven-development`: for each task in the issue, write a failing test first, then implement. For bug fixes, `superpowers:systematic-debugging` first.
   2. Follow the issue's referenced OpenSpec design/tasks files; tick the task in `tasks.md` if referenced.
   3. Use `superpowers:verification-before-completion`: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -m "not live"` must pass.
   4. Commit with a message ending in `Closes #<n>` and the `Co-Authored-By` trailer.
   5. Write `.run_issue/PR_BODY.md` with headings `## Summary`, `## Implementation details`, `## In plain words`, `## Verification`, `## Routing`.
   6. End with a final line `RESULT: done` or `RESULT: blocked: <reason>`.

4. **Post-checks** (runner):
   - agent output's result text ends with `RESULT: done`;
   - `git rev-list origin/<base>..HEAD` non-empty;
   - `PR_BODY.md` exists and has all five headings;
   - `git diff --name-only origin/<base>..HEAD` includes a path under `tests/`;
   - runner re-runs ruff check, ruff format --check, pytest `-m "not live"` in the worktree.

5. **Failure handling:** any post-check failure → one fix round (`claude -p` with `fix.md`, same model and allowlist, containing the failed check's output tail, last 200 lines). Still failing and model is Sonnet → one retry round on Opus (model-routing escalation rule). Still failing → outcome `blocked`, worktree kept, **run halts** (later issues may depend on this one). `RESULT: blocked` from the agent halts immediately without fix rounds.

## Phase 4 — PR, CI, merge

1. `git push -u origin <branch>`.
2. If `gh pr list --head <branch> --state open` finds a PR, reuse it (and `gh pr edit --body-file`); else `gh pr create --base <base> --head <branch> --title "<title without tier tag> (#<n>)" --body-file <body>` where body = `PR_BODY.md` + `\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
3. Wait for checks to register: poll `gh pr checks <pr>` every 10 s up to 5 min. No checks after that → outcome `ci-failed` (reason "no CI checks reported"), not merged.
4. `gh pr checks <pr> --watch --fail-fast`.
   - exit 0 → `gh pr merge <pr> --squash --delete-branch`; outcome `merged`.
   - non-zero → fetch failed log: newest failed run on the branch via `gh run list --branch <branch> --status failure --limit 1 --json databaseId`, then `gh run view <id> --log-failed | tail -200`. Agent fix round with `fix.md` (CI variant, instructs `superpowers:systematic-debugging`), then runner local re-verify, push, back to step 3. Up to `--max-ci-fixes` rounds.
   - exhausted → `gh pr comment` summarizing the failure, PR left open, outcome `ci-failed`, **run halts**.

## Phase 5 — Bookkeeping

After a merge:
- Parent issue (from `gh api repos/{repo}/issues/{n}/parent`, if any): fetch body fresh, replace `- [ ] ` with `- [x] ` on the single line containing `(#<n>)`, `gh issue edit <parent> --body-file`. No change if no matching unchecked line.
- `gh issue comment <n> --body "Merged in <pr-url>"` (issue closes via `Closes #n`).
- `git worktree remove` and delete local branch.

## Logs, summary, exit codes

- Logs: `.claude/worktrees/run-logs/<UTC timestamp>/issue-<n>.log` (runner log + each `claude` JSON output).
- Final summary table on stdout: issue, model, PR URL, outcome (`merged` / `skipped-human` / `skipped-closed` / `blocked` / `ci-failed` / `planned` for dry-run), and `total_cost_usd` / `num_turns` summed from `claude` JSON where present.
- Exit codes: `0` all implementable issues merged (skips are fine); `1` run halted; `2` usage/precondition error.

## Files

```
scripts/run_issue.sh                 entrypoint: arg parsing, preconditions, main loop
scripts/run_issue/lib.sh             functions: parse_tier, strip_tier, slugify, expand_issues,
                                     route_issue, check_pr_body, tick_parent_line, extract_json …
scripts/run_issue/implement.md       implementation prompt template
scripts/run_issue/fix.md             fix prompt template (local / CI)
scripts/run_issue/route.md           routing prompt template
scripts/run_issue/allowed_tools.txt  allowlist
tests/test_run_issue.py              pytest suite
.gitignore                           + .run_issue/
```

Templates use `{{PLACEHOLDER}}` substitution done in bash (no extra deps). `gh --jq` is used for JSON; no `jq` dependency.

## Testing

`tests/test_run_issue.py`, runs in existing CI (`pytest -m "not live"`), no network, no real Claude.

- **Unit** (source `lib.sh` via `bash -c`): tier parsing from title / body / none; strip_tier + slugify; Fable gate with and without `--allow-fable`; sub-issue expansion, dedupe, closed-skip, `(#N)` fallback; `tick_parent_line` ticks only the matching line; `check_pr_body` heading validation; routing JSON extraction and Sonnet fallback.
- **End-to-end with fakes**: temp bare repo as `origin`, a clone as the working repo; fake `claude`, `gh`, `uv` executables on `PATH` that log argv/env to a file and replay scripted responses from a scenario JSON. Scenarios:
  - happy path: `--model sonnet`, allowlist passed, PR created with body file, merge called, parent ticked;
  - `[Human]` issue skipped, no `claude` call;
  - CI fails once → fix prompt sent → merged;
  - CI fails beyond `--max-ci-fixes` → no merge, exit 1, later issue not started;
  - no checks reported → no merge;
  - `--dry-run` → no worktree/push/PR;
  - `ANTHROPIC_API_KEY` absent from fake `claude`'s environment.
- **Lint**: `shellcheck` on both scripts if available, test skipped otherwise.

## Non-goals

Parallel issue execution; persistent resume state (rerun reuses branch/worktree/PR); non-GitHub remotes; model aliases beyond opus/sonnet/fable; implementing `[Human]` issues.
