#!/bin/zsh
# Phase 1 — honest baseline. Runs the "where does the agent stand TODAY?" suite and
# saves every report into one timestamped folder so you can compare against it later.
#   ./scripts/run_baseline.sh            # 30-episode tempered eval (default)
#   ./scripts/run_baseline.sh 50         # custom episode count
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
EPISODES="${1:-30}"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="./reports/baseline-$STAMP"
mkdir -p "$OUT"
echo "================ BASELINE → $OUT ================"

# Keep the SQLite view fresh so timeline/db reflect reality.
$PY doom_cli.py db build > /dev/null 2>&1 || true

echo "---- timeline (evolution so far) ----"
$PY doom_cli.py timeline | tee "$OUT/timeline.txt"

echo "---- eval (tempered T=0.5, $EPISODES episodes) — the honest metrics ----"
$PY doom_cli.py eval --temperature 0.5 --episodes "$EPISODES" --json | tee "$OUT/baseline.json"

echo "---- audit (is it REALLY learning?) ----"
$PY doom_cli.py audit | tee "$OUT/audit.txt"

echo "---- behavior (circling / passive / low-exploration?) ----"
$PY doom_cli.py behavior | tee "$OUT/behavior.txt"

echo ""
echo "================ BASELINE DONE ================"
echo "Reports saved in: $OUT"
echo "  baseline.json  <- compare future runs against THIS"
echo "  timeline.txt   audit.txt   behavior.txt"
echo ""
echo "Next (Phase 2, highest ROI): record your own play, clone it, re-eval:"
echo "  python scripts/record_demo.py --map MAP01 --episodes 3 --strafe --minutes 10"
echo "  doom-cli bc --epochs 10"
echo "  doom-cli eval --temperature 0.5 --episodes $EPISODES --json"
