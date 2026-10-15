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
