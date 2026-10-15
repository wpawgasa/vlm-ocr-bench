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
