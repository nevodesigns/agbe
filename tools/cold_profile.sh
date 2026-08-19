#!/usr/bin/env bash
# Cold-machine profiler run. Do this AFTER a restart with nothing else open.
#
# The thermal penalty is 10 points, a sixth of the engineering score, and we have
# measured 98 to 99 degrees under sustained back-to-back load on this laptop. So
# the run wants a genuinely cold start: restart, no browser, no editor, no Kaggle
# tab, laptop elevated, fan on.
#
# This script deliberately does NOT loop waiting for the machine to cool. An
# earlier version did, reading the max across every hwmon sensor, and acpitz idles
# near 70C on this machine, so the condition was never satisfiable and the
# profiler simply never started for 36 minutes. It reads coretemp only, reports
# it, and leaves the judgement to you.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILER=/home/nwokolo/projects/adtc-2026/.venv/bin/adtc-profiler
MODEL=model/agbe-1b-q4_k_m.gguf

# The profiler shells out to `llama-bench` and only looks on PATH. Ours is built
# in the working repo rather than installed system-wide, so put it there. Without
# this the run dies with "llama-bench not found on PATH" AFTER printing that it
# has started, which reads like the profiler is broken rather than the PATH.
LLAMA_BIN=/home/nwokolo/projects/adtc-2026/work/llama.cpp/build/bin
export PATH="$LLAMA_BIN:$PATH"
command -v llama-bench > /dev/null || {
  echo "llama-bench still not found. Expected it at $LLAMA_BIN"; exit 1; }
echo "llama-bench: $(command -v llama-bench)"

[ -x "$PROFILER" ] || { echo "profiler not found at $PROFILER"; exit 1; }
[ -f "$MODEL" ]    || { echo "no weights at $MODEL"; exit 1; }

echo "weights : $(du -h "$MODEL" | cut -f1)  sha256 $(sha256sum "$MODEL" | cut -c1-16)…"

# coretemp only. Taking a max across every sensor sweeps in chipset, wifi and
# ACPI zones that sit far above the CPU package at idle.
for hw in /sys/class/hwmon/hwmon*; do
  [ "$(cat "$hw/name" 2>/dev/null)" = "coretemp" ] || continue
  t=$(cat "$hw"/temp*_input 2>/dev/null | sort -n | tail -1)
  printf "cpu     : %d.%d C" $((t/1000)) $(((t%1000)/100))
  if [ "$t" -gt 60000 ]; then
    echo "   ← WARM. Wait a few minutes; a hot start costs 10 points."
  else
    echo "   ← cold, good to go"
  fi
done

echo
echo "running profiler, this takes a few minutes and the machine will get busy…"
"$PROFILER" run --submission . --mode participant --output submission.json

echo
python3 - <<'PY'
import json
d = json.load(open("submission.json"))
tps  = d["throughput"]["tokens_per_second_generation"]
rss  = d["memory"]["peak_rss_mb"]
therm = d.get("thermal", {})
s_perf = min(tps / 15.0, 1.0) * 100
s_eff  = max(0.0, (7.0 - rss / 1024) / 7.0) * 100
print(f"  throughput   {tps:.2f} tok/s      -> S_perf {s_perf:.1f}/100")
print(f"  peak RSS     {rss:.0f} MB         -> S_eff  {s_eff:.1f}/100")
print(f"  thermal      {therm if therm else 'no block reported'}")
print(f"\n  engineering  {(0.30*s_perf + 0.20*s_eff):.2f} of the 50 available")
print(f"  team_id      {d['submission']['team_id']}")
for p in d["submission"]["test_prompts"]:
    print(f"  {p['prompt_id']}        {p['prompt'][:66]}…")
PY
