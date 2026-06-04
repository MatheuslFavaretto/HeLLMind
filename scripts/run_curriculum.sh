#!/bin/zsh
# Progressive RL curriculum (V2 — the biggest V1 miss):
#   Stage 1: my_way_home  — pure navigation (no enemies, just find the exit)
#   Stage 2: deadly_corridor — navigation + survive enemies (health focus)
#   Stage 3: MAP01 (full)  — the real game
#
# The V1 mistake was training directly on full maps (hard). This teaches the individual
# skills first, then combines them — the approach that gets exit-rate off zero.
#
#   ./scripts/run_curriculum.sh              # default steps per stage
#   ./scripts/run_curriculum.sh 200000      # custom steps per stage
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
STEPS="${1:-150000}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="./reports/curriculum-$STAMP"
mkdir -p "$OUT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  HeLLMind Progressive Curriculum (V2)                       ║"
echo "║  Stage 1 → my_way_home (navigation)                         ║"
echo "║  Stage 2 → deadly_corridor (navigation + survival)          ║"
echo "║  Stage 3 → MAP01 full (the real thing)                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  steps/stage: $STEPS  |  reports → $OUT"
echo ""

# ── Stage 1: my_way_home (navigation only, no combat reward pressure) ──────────
echo "══ Stage 1/3: my_way_home — learn to find the exit ══"
CAMPAIGN=0 \
DOCS_ENABLED=0 \
MEMORY_ENABLED=0 \
COVERAGE_REWARD=1.0 \
FRONTIER_REWARD=0.1 \
EXIT_REWARD=500.0 \
HIT_REWARD=0 MISS_PENALTY=0 DEATH_PENALTY=1.0 \
ENGAGEMENT_REWARD=0 BESTIARY_REWARD=0 \
EPISODE_TIMEOUT=2100 \
DOOM_SCENARIO=my_way_home $PY -m rl.train --fresh --timesteps "$STEPS"

echo ""
echo "── Stage 1 eval ──"
CAMPAIGN=0 DOOM_SCENARIO=my_way_home $PY -m rl.eval --episodes 20 --json | tee "$OUT/stage1_eval.json"

# ── Stage 2: deadly_corridor (navigation + don't die) ──────────────────────────
echo ""
echo "══ Stage 2/3: deadly_corridor — learn to survive + navigate ══"
CAMPAIGN=0 \
DOCS_ENABLED=0 \
MEMORY_ENABLED=0 \
COVERAGE_REWARD=0.5 \
FRONTIER_REWARD=0.05 \
EXIT_REWARD=200.0 \
DEATH_PENALTY=10.0 \
DAMAGE_TAKEN_PENALTY=0.3 \
EPISODE_TIMEOUT=2100 \
DOOM_SCENARIO=deadly_corridor $PY -m rl.train --timesteps "$STEPS"

echo ""
echo "── Stage 2 eval ──"
CAMPAIGN=0 DOOM_SCENARIO=deadly_corridor $PY -m rl.eval --episodes 20 --json | tee "$OUT/stage2_eval.json"

# ── Stage 3: full MAP01 (transfer what was learned) ────────────────────────────
echo ""
echo "══ Stage 3/3: MAP01 (full game — the real thing) ══"
CAMPAIGN=1 \
DOCS_ENABLED=0 \
MEMORY_ENABLED=1 \
$PY -m rl.train --maps MAP01 --timesteps "$STEPS"

echo ""
echo "── Stage 3 eval ──"
$PY doom_cli.py eval --episodes 20 --json | tee "$OUT/stage3_eval.json"

echo ""
echo "══ CURRICULUM DONE — reports in $OUT ══"
$PY doom_cli.py timeline
