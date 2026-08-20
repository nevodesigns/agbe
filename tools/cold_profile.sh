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

# coretemp only, sampled over time rather than read once.
#
# Two separate mistakes are corrected here. First, an earlier version took the max
# across every hwmon sensor, and acpitz idles near 73C on this machine, so the
# reading was never usable. Second, and less obvious: core temperature swings
# several degrees between consecutive reads, so a single instantaneous sample
# catches spikes. This script reported 86C on a machine whose cores were sitting
# at 62 to 71C, and told the user to go and cool a laptop that was already cool.
#
# Sample for a few seconds and use the MEDIAN. Same fix as the continuous sampler
# in work/bench.py, which was written for exactly this reason and not reused here.
CORE_HW=""
for hw in /sys/class/hwmon/hwmon*; do
  [ "$(cat "$hw/name" 2>/dev/null)" = "coretemp" ] && CORE_HW="$hw" && break
done
[ -n "$CORE_HW" ] || { echo "no coretemp sensor found"; exit 1; }

samples=""
for i in $(seq 1 12); do
  v=$(cat "$CORE_HW"/temp*_input 2>/dev/null | sort -n | tail -1)
  samples="$samples $v"
  sleep 0.25
done
t=$(echo $samples | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')
peak=$(echo $samples | tr ' ' '\n' | sort -n | tail -1)
printf "cpu     : %d.%d C median, %d.%d C peak over 3s" \
  $((t/1000)) $(((t%1000)/100)) $((peak/1000)) $(((peak%1000)/100))

# Report what is actually generating heat, since that is the actionable part.
busy=$(ps -eo pcpu,comm --sort=-pcpu | awk 'NR>1 && $1>5 {printf "%s(%.0f%%) ", $2, $1}' | head -c 120)

if [ "$t" -gt 75000 ]; then
  echo "   <- TOO HOT to start; this run would throttle and cost 10 points."
  [ -n "$busy" ] && echo "busy    : $busy"
  echo "Close those, wait for the median to drop under 65C, and re-run."
  echo "To profile anyway and accept the penalty: AGBE_FORCE_HOT=1 $0"
  [ "${AGBE_FORCE_HOT:-0}" = "1" ] || exit 2
elif [ "$t" -gt 65000 ]; then
  echo "   <- warm but workable."
  [ -n "$busy" ] && echo "busy    : $busy   <- closing these would help"
else
  echo "   <- cold, good to go"
fi

echo
echo "running profiler, this takes a few minutes and the machine will get busy…"
"$PROFILER" run --submission . --mode participant --output submission.json

echo
python3 - <<'PY'
import json
d = json.load(open("submission.json"))
tps  = d["throughput"]["tokens_per_second_generation"]
rss  = d["memory"]["peak_rss_mb"]
therm = d.get("cpu_thermal", {})
s_perf = min(tps / 15.0, 1.0) * 100
s_eff  = max(0.0, (7.0 - rss / 1024) / 7.0) * 100
print(f"  throughput   {tps:.2f} tok/s      -> S_perf {s_perf:.1f}/100")
print(f"  peak RSS     {rss:.0f} MB         -> S_eff  {s_eff:.1f}/100")
# Read cpu_thermal, not thermal. The first version of this script looked for the
# wrong key, printed "no block reported", and hid the fact that the run had
# throttled at 99C: it reported 47.10 when the honest number was 37.10.
peak = therm.get("core_temp_c_peak")
thr  = therm.get("throttled")
print(f"  peak temp    {peak} C   throttled={thr}")
pen = -10.0 if (thr or (peak or 0) > 85) else 0.0
eng = 0.30 * s_perf + 0.20 * s_eff
print(f"\n  engineering  {eng:.2f} of the 50 available")
if pen:
    print(f"  THERMAL      {pen:.0f}  (throttled or over 85C)")
    print(f"  net          {eng + pen:.2f}   <-- re-run cold to recover this")
print(f"  team_id      {d['submission']['team_id']}")
for p in d["submission"]["test_prompts"]:
    print(f"  {p['prompt_id']}        {p['prompt'][:66]}…")
PY
