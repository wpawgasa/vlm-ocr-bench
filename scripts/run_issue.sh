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
  local n=$1 title=$2 body=$3 model=$4 reason=$5 prompt status failure m top cur reused=0
  ISSUE_BRANCH=$(branch_name "$n" "$title")
  ISSUE_WT="$ROOT/.claude/worktrees/issue-$n"
  ISSUE_MODEL=$model
  git -C "$ROOT" fetch -q origin >>"$CURRENT_LOG" 2>&1 || {
    log "#$n: git fetch failed"
    return 1
  }
  if [[ -d $ISSUE_WT ]]; then
    reused=1
    top=$(git -C "$ISSUE_WT" rev-parse --show-toplevel 2>/dev/null)
    if [[ -z $top || $(cd "$top" && pwd -P) != "$(cd "$ISSUE_WT" && pwd -P)" ]]; then
      log "#$n: $ISSUE_WT exists but is not a git worktree; remove it and re-run"
      return 1
    fi
    cur=$(git -C "$ISSUE_WT" symbolic-ref --short HEAD 2>/dev/null) || {
      log "#$n: worktree $ISSUE_WT has a detached HEAD; check out a branch or remove it"
      return 1
    }
    if [[ $cur != "$ISSUE_BRANCH" ]]; then
      log "#$n: worktree is on $cur, not $ISSUE_BRANCH (issue retitled?); using $cur"
      ISSUE_BRANCH=$cur
    fi
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
        if ! gh pr merge "$pr" -R "$REPO" --squash --delete-branch \
          --match-head-commit "$sha" >>"$CURRENT_LOG" 2>&1; then
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
  ship_issue "$n" "$title"
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
