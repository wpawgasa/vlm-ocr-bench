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
  implement_issue "$n" "$title" "$body" "$model" "$reason" || return 1
  OUT_OUTCOME[$n]=implemented
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
