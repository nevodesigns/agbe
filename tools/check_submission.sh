#!/usr/bin/env bash
# Pre-submission consistency check.
#
# Exists because a stale DEVPOST_STORY.md survived the Pidgin withdrawal and
# claimed a capability that metadata.json and REPORT.md both denied. Two files
# in one repo disagreeing about what the model does is worse than the weakness
# being disclosed at all.
set -u
cd "$(dirname "$0")/.."
fail=0
note() { printf "  %-6s %s\n" "$1" "$2"; }

if grep -rn "REPLACE_WITH\|your-team-id" metadata.json REPORT.md 2>/dev/null | grep -q .; then
  note FAIL "placeholder still present:"
  grep -rn "REPLACE_WITH\|your-team-id" metadata.json REPORT.md | sed 's/^/         /'
  fail=1
else
  note ok "no placeholders in metadata.json or REPORT.md"
fi

scope=$(python3 -c "import json;print(','.join(json.load(open('metadata.json'))['language_scope']))")
if [ "$scope" = "en" ] && grep -rqi "in english and in nigerian pidgin" --include="*.md" . 2>/dev/null; then
  note FAIL "language_scope is [$scope] but a doc still claims Pidgin support"
  fail=1
else
  note ok "language_scope [$scope] agrees with the docs"
fi

n=$(find . -iname "devpost*story*.md" -not -path "./.git/*" | wc -l)
if [ "$n" -le 1 ]; then note ok "one Devpost story file"
else note FAIL "$n Devpost story files, they will drift apart"; fail=1; fi

python3 - <<'PY' || fail=1
import json, pathlib, sys
m = json.load(open("metadata.json"))
missing = [p["prompt"] for p in m["test_prompts"]
           if p["prompt"][:60] not in pathlib.Path("REPORT.md").read_text()]
print("  ok     both test prompts appear in REPORT.md" if not missing
      else f"  FAIL   {len(missing)} test prompt(s) missing from REPORT.md")
sys.exit(1 if missing else 0)
PY

# HTTP 200 is not enough. A cancelled Kaggle run once uploaded a 15 MB partial
# GGUF over the canonical filename, and this check happily reported READY TO
# SUBMIT while the public URL served a broken model. Verify the published file is
# the exact artifact download_model.sh pins.
URL="https://huggingface.co/NEVODESIGN/agbe-1b/resolve/main/agbe-1b-q4_k_m.gguf"
PIN_SHA=$(grep '^EXPECT_SHA256=' download_model.sh | cut -d= -f2)
PIN_LEN=$(grep '^EXPECT_BYTES=' download_model.sh | cut -d= -f2)
hdr=$(curl -sIL "$URL" 2>/dev/null | tr -d '\r')
code=$(curl -sI -L -o /dev/null -w "%{http_code}" "$URL")
rem_sha=$(printf '%s\n' "$hdr" | grep -i '^x-linked-etag' | tr -d '"' | awk '{print $2}' | tail -1)
rem_len=$(printf '%s\n' "$hdr" | grep -i '^x-linked-size' | awk '{print $2}' | tail -1)

if [ "$code" != "200" ]; then
  note FAIL "weights not publicly fetchable (HTTP $code)"; fail=1
elif [ "$rem_len" != "$PIN_LEN" ]; then
  note FAIL "published weights are $rem_len bytes, pinned expects $PIN_LEN"; fail=1
elif [ -n "$rem_sha" ] && [ "$rem_sha" != "$PIN_SHA" ]; then
  note FAIL "published sha256 ${rem_sha:0:16}… does not match pinned ${PIN_SHA:0:16}…"; fail=1
else
  note ok "weights public and match the pinned sha256 ($PIN_LEN bytes)"
fi

if [ -f submission.json ]; then note ok "submission.json telemetry present"
else note FAIL "submission.json missing"; fail=1; fi

# Documents, not just the artifact. The artifact checks below proved the model's
# identity while README, REPORT, BUILDS and the site quietly drifted apart, and
# this script still printed READY TO SUBMIT. FINAL.json is now the single source
# of truth and tools/check_docs.py rejects any public file that contradicts it.
if python3 "$(dirname "$0")/check_docs.py" > /tmp/agbe-docs.$$ 2>&1; then
  echo "  ok     public documents agree with FINAL.json"
else
  echo "  FAIL   public documents contradict FINAL.json:"
  sed 's/^/  /' /tmp/agbe-docs.$$ | head -20
  fail=1
fi
rm -f /tmp/agbe-docs.$$

# The build ledger is a second kind of drift: FINAL.json only describes the shipped
# model, so a stale historical row in BUILDS.md passes check_docs.py while
# contradicting the stored eval answers. It did, for three rows.
if python3 "$(dirname "$0")/ledger.py" --check > /tmp/agbe-ledger.$$ 2>&1; then
  echo "  ok     build ledger agrees with the stored eval answers"
else
  echo "  FAIL   BUILDS.md contradicts eval/*.json:"
  sed 's/^/  /' /tmp/agbe-ledger.$$ | tail -10
  fail=1
fi
rm -f /tmp/agbe-ledger.$$

# check_docs.py bans strings already known to be wrong, which only ever catches
# a mistake after a human has found it. It caught neither of the two that
# mattered: "104 optimiser steps" wrapped across a line break, and "13.0M
# trainable" was simply never on the banned list. consistency.py inverts the
# test and asserts the manifest's values against every number stated in their
# context, so it fails on wrong values nobody has seen yet.
if python3 "$(dirname "$0")/consistency.py" > /tmp/agbe-consist.$$ 2>&1; then
  echo "  ok     no document number contradicts FINAL.json"
else
  echo "  FAIL   a document number contradicts FINAL.json:"
  sed 's/^/  /' /tmp/agbe-consist.$$ | tail -12
  fail=1
fi
rm -f /tmp/agbe-consist.$$

# --- Gate 2, section 3 ------------------------------------------------------
#
# The semifinalist round added requirements that the Round 1 checks knew nothing
# about. These are mechanical, so they are checked rather than remembered.
echo
echo "  Gate 2 (semifinalist round)"

R="$(dirname "$0")/.."

# 3.1 the git commit SHA lives in metadata.json
if bash "$(dirname "$0")/lock_commit.sh" --check > /dev/null 2>&1; then
  echo "  ok     3.1  metadata.json git_commit_sha matches HEAD"
else
  echo "  WARN   3.1  metadata.json git_commit_sha is not HEAD"
  echo "              run tools/lock_commit.sh as the last step before submitting"
fi

# 3.1 the Model Provenance section exists and says the required things
missing=""
for needle in "Base model and exact source" "Fine-tuning method"               "Training dataset" "Before and after"; do
  grep -qi "$needle" "$R/REPORT.md" || missing="$missing '$needle'"
done
if [ -z "$missing" ]; then
  echo "  ok     3.1  REPORT.md model provenance section covers all four items"
else
  echo "  FAIL   3.1  REPORT.md model provenance is missing:$missing"
  fail=1
fi

# 3.1 the proof-of-training bundle
for f in provenance/README.md provenance/merge_and_quantise.sh          provenance/compare_to_base.sh provenance/before-after.md; do
  if [ -f "$R/$f" ]; then echo "  ok     3.1  $f"
  else echo "  FAIL   3.1  missing $f"; fail=1; fi
done
for f in provenance/adapter/adapter_model.safetensors          provenance/adapter/adapter_config.json          provenance/training_log.json provenance/run_manifest.json          provenance/checksums.json; do
  if [ -f "$R/$f" ]; then echo "  ok     3.1  $f"
  else
    echo "  TODO   3.1  $f not yet present"
    echo "              produced by the Kaggle run; unpack provenance.zip here"
    fail=1
  fi
done

# 3.2 the download URL must be readable without executing the script
url_lines=$(grep -c '^MODEL_URL="https://[^"$]*"$' "$R/download_model.sh" || true)
if [ "$url_lines" -eq 1 ] && ! grep -qE '^MODEL_URL=.*\$\{' "$R/download_model.sh"; then
  echo "  ok     3.2  download_model.sh URL is a static literal"
else
  echo "  FAIL   3.2  download_model.sh URL is not a plain static string"
  fail=1
fi

# 3.1 metadata.json must still satisfy the profiler's own validator. Adding a
# plain `reproducibility` key, which section 3.1 asks for in so many words, makes
# adtc-profiler exit 2 before it benchmarks anything.
PV=/home/nwokolo/projects/adtc-2026/.venv/bin/python
if [ -x "$PV" ]; then
  if "$PV" - "$R/metadata.json" <<'PYEOF' > /dev/null 2>&1
import json, sys
from adtc_profiler import report
m = json.load(open(sys.argv[1]))
report.validate_submission_block({k: v for k, v in m.items() if not k.startswith("_")})
PYEOF
  then echo "  ok     3.1  metadata.json passes the profiler's own schema validator"
  else echo "  FAIL   3.1  adtc-profiler would REJECT metadata.json (it exits 2 before running)"; fail=1; fi
else
  echo "  skip   3.1  profiler venv not found, cannot validate metadata.json"
fi

# 3.4 self-reported figures have to come from a committed profiler run
if [ -f "$R/submission.json" ]; then
  echo "  ok     3.4  submission.json committed for independent comparison"
else
  echo "  FAIL   3.4  submission.json missing"; fail=1
fi

echo

if [ "$fail" -eq 0 ]; then echo "  READY TO SUBMIT"; else echo "  NOT READY: fix the FAIL lines above"; fi
exit $fail
