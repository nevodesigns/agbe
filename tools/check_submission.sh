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

code=$(curl -sI -L -o /dev/null -w "%{http_code}" \
  "https://huggingface.co/NEVODESIGN/agbe-1b/resolve/main/agbe-1b-q4_k_m.gguf")
if [ "$code" = "200" ]; then note ok "weights public (HTTP 200, no credentials)"
else note FAIL "weights not publicly fetchable (HTTP $code)"; fail=1; fi

if [ -f submission.json ]; then note ok "submission.json telemetry present"
else note FAIL "submission.json missing"; fail=1; fi

echo
if [ "$fail" -eq 0 ]; then echo "  READY TO SUBMIT"; else echo "  NOT READY: fix the FAIL lines above"; fi
exit $fail
