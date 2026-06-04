#!/bin/zsh
# Phase 4 — pay down the validation debt. The project's rule: a feature only stays ON if it
# PROVES it helps on honest eval. This runs the falsifiable loop end to end:
#   behavior flags -> hypothesis -> multi-seed A/B (control vs experimental) -> honest verdict.
#
#   ./scripts/run_phase4.sh                 # auto-pick the top open hypothesis, 200k steps
#   ./scripts/run_phase4.sh 3 150000        # specific hypothesis id, custom steps/arm
#
# Heavy: it trains TWO branches × seeds. Best left running overnight on a laptop.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
HYP_ID="${1:-}"
STEPS="${2:-200000}"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="./reports/phase4-$STAMP"
mkdir -p "$OUT"
echo "================ PHASE 4 (A/B validation) → $OUT ================"

# Keep the read-view fresh so behavior/hypothesize see the latest telemetry.
$PY doom_cli.py db build > /dev/null 2>&1 || true

echo "---- 1/3  HYPOTHESIZE: turn behavior flags into falsifiable hypotheses ----"
$PY doom_cli.py hypothesize | tee "$OUT/hypotheses.txt"

# Pick the hypothesis to test: the explicit arg, else the first OPEN one.
if [ -z "$HYP_ID" ]; then
  HYP_ID=$($PY - <<'PYEOF'
from config import Config
from writer.db import query_hypotheses
rows = query_hypotheses(Config().memory_dir, status="open")
print(rows[0]["id"] if rows else "")
PYEOF
)
fi

if [ -z "$HYP_ID" ]; then
  echo "No open hypotheses to test. (Run more training so behavior flags appear, then retry.)"
  exit 0
fi

echo "---- 2/3  EXPERIMENT: A/B hypothesis #$HYP_ID ($STEPS steps/arm, seeds 42,123) ----"
$PY doom_cli.py experiment --hypothesis "$HYP_ID" --steps "$STEPS" --seeds 42,123 --episodes 15 \
  | tee "$OUT/experiment.txt"

echo "---- 3/3  VERDICT: what the A/B recorded ----"
$PY doom_cli.py db build > /dev/null 2>&1 || true
$PY doom_cli.py db query --experiments | tee "$OUT/verdict.txt"

echo ""
echo "================ PHASE 4 DONE ================"
echo "Reports saved in: $OUT"
echo "  A verdict of 'improved' is auto-adopted into learned_config (persists across runs)."
echo "  Check it stuck:  doom-cli learned"
