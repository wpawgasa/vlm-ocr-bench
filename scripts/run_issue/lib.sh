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
    "RESULT: done"*) echo "done" ;;
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

# protected_hits PREFIX_FILE — read file names on stdin; print each one that
# starts with a prefix listed in PREFIX_FILE (one per line, comments allowed).
protected_hits() {
  local -a prefixes
  local name prefix
  mapfile -t prefixes < <(load_tools "$1")
  while IFS= read -r name; do
    for prefix in "${prefixes[@]}"; do
      if [[ -n $name && $name == "$prefix"* ]]; then
        printf '%s\n' "$name"
        break
      fi
    done
  done
}

# Credentials the agent and the checks run on its code must not see.
AGENT_SECRETS=(
  ANTHROPIC_API_KEY
  ANTHROPIC_AUTH_TOKEN
  GH_TOKEN
  GITHUB_TOKEN
  GH_ENTERPRISE_TOKEN
  SSH_AUTH_SOCK
)

# agent_env GH_CONFIG_DIR — the `env` arguments, one per line, for running agent
# code: credential variables unset (Claude then uses the subscription login), gh
# pointed at the empty directory GH_CONFIG_DIR, git prompts and credential
# helpers disabled. Use: mapfile -t ENV < <(agent_env DIR); env "${ENV[@]}" CMD.
agent_env() {
  local name
  for name in "${AGENT_SECRETS[@]}"; do
    printf -- '-u%s\n' "$name"
  done
  printf '%s\n' "GH_CONFIG_DIR=$1" GIT_TERMINAL_PROMPT=0 GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0=
}

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
