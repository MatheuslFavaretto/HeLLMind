#!/bin/zsh
# Phase 2 — learn from YOU (behavioral cloning), then measure the lift vs the baseline.
# This is the highest-ROI lever for the hardest problem (reaching the exit): it turns
# "explore a maze blindly" into "fine-tune something that already roughly works".
#
#   ./scripts/run_phase2.sh                 # MAP01, 3 demo episodes, 30-episode eval
#   ./scripts/run_phase2.sh MAP01 5 50      # map, demo episodes, eval episodes
#
# Recording needs a game WINDOW (play on your Mac, not headless Colab). Play to the EXIT —
# those are the demos worth recording. Close the window any time to stop; episodes are kept.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
MAP="${1:-MAP01}"
DEMO_EPS="${2:-3}"
EVAL_EPS="${3:-30}"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="./reports/phase2-$STAMP"
mkdir -p "$OUT"
echo "================ PHASE 2 (BC) → $OUT ================"

echo "---- 1/3  RECORD: you play $MAP to the EXIT ($DEMO_EPS episode(s)) ----"
echo "      (close the game window to stop early — recorded episodes are kept)"
$PY scripts/record_demo.py --map "$MAP" --episodes "$DEMO_EPS" --strafe --minutes 10

echo "---- 2/3  CLONE: the agent imitates your play ----"
$PY doom_cli.py bc --epochs 10 | tee "$OUT/bc.txt"

echo "---- 3/3  EVAL (tempered T=0.5, $EVAL_EPS episodes) ----"
$PY doom_cli.py eval --temperature 0.5 --episodes "$EVAL_EPS" --json | tee "$OUT/after_bc.json"

# Compare against the most recent baseline, if one exists.
BASELINE=$(ls -dt ./reports/baseline-*/baseline.json 2>/dev/null | head -1 || true)
echo ""
echo "================ LIFT vs BASELINE ================"
if [ -n "$BASELINE" ]; then
  echo "baseline: $BASELINE"
  $PY - "$BASELINE" "$OUT/after_bc.json" <<'PYEOF'
import json, sys

def load(path):
    # eval prints "METRICS_JSON {...}" mixed with human text — pull the last such line.
    with open(path, encoding="utf-8") as f:
        line = [l for l in f if l.strip().startswith("METRICS_JSON")]
    return json.loads(line[-1].split("METRICS_JSON", 1)[1]) if line else {}

before, after = load(sys.argv[1]), load(sys.argv[2])
keys = ["exit_rate", "explored_fraction", "kills_per_episode",
        "shooting_accuracy", "death_rate", "timeout_rate"]
pct = {"exit_rate", "explored_fraction", "shooting_accuracy", "death_rate", "timeout_rate"}
print(f"{'metric':20} {'before':>10} {'after':>10} {'delta':>10}")
for k in keys:
    b, a = before.get(k), after.get(k)
    if b is None or a is None:
        continue
    d = a - b
    fmt = (lambda x: f"{x*100:.1f}%") if k in pct else (lambda x: f"{x:.2f}")
    arrow = "↑" if d > 1e-6 else ("↓" if d < -1e-6 else "→")
    print(f"{k:20} {fmt(b):>10} {fmt(a):>10} {arrow} {fmt(abs(d)):>8}")
if after.get("exit_rate", 0) > 0:
    print("\n*** EXIT REACHED — exit_rate > 0! This is the milestone. ***")
PYEOF
else
  echo "No baseline found. Run ./scripts/run_baseline.sh first to compare against."
fi
echo ""
echo "Reports saved in: $OUT"
