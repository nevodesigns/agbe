#!/usr/bin/env bash
# Stamp the current HEAD into metadata.json.
#
# Gate 2 section 3.1: "The Git Commit SHA must be added to the metadata.json."
#
# A commit cannot contain its own SHA, so this is inherently one step behind:
# run it as the LAST action before submitting, commit the one-line change, and
# the recorded SHA names the tree immediately before that final commit. Anything
# else is a claim you cannot keep true.
#
#   bash tools/lock_commit.sh          # stamp HEAD
#   bash tools/lock_commit.sh --check  # verify it matches HEAD, exit 1 if not

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HEAD_SHA="$(git rev-parse HEAD)"
CURRENT="$(python3 -c 'import json;print(json.load(open("metadata.json")).get("_reproducibility",{}).get("git_commit_sha",""))')"

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$CURRENT" == "$HEAD_SHA" ]]; then
    echo "metadata.json git_commit_sha matches HEAD ($HEAD_SHA)"
    exit 0
  fi
  echo "metadata.json git_commit_sha is $CURRENT but HEAD is $HEAD_SHA" >&2
  echo "run: bash tools/lock_commit.sh && git commit -am 'stamped the submission commit'" >&2
  exit 1
fi

python3 - "$HEAD_SHA" <<'PY'
import json, pathlib, sys
sha = sys.argv[1]
p = pathlib.Path("metadata.json")
m = json.loads(p.read_text())
m.setdefault("_reproducibility", {})["git_commit_sha"] = sha
p.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
print(f"metadata.json git_commit_sha -> {sha}")
PY
