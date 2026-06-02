#!/bin/zsh
# A/B: current reward shaping vs combat-retuned shaping. Fresh MAP01, same seed.
set -e
cd /Users/matheusfavaretto/Documents/labs/poc-doom-obisidyan
PY=.venv/bin/python
STEPS=80000

# Shared, isolated-from-vault settings (no Ollama, no memory side-effects).
export CAMPAIGN=1 MAPS=MAP01 DOCS_ENABLED=0 MEMORY_ENABLED=0 N_ENVS=4 SEED=42

run() {  # $1=label  $2=ckptdir  then reward env already exported
  echo "================ TRAIN $1 ================"
  rm -rf "$2"; mkdir -p "$2"
  CHECKPOINT_DIR="$2" RUN_NAME="ab-$1" \
    $PY -m rl.train --fresh --maps MAP01 --n-envs 4 --timesteps $STEPS
  echo "================ EVAL  $1 ================"
  CHECKPOINT_DIR="$2" $PY -m rl.eval --episodes 10 \
    --path "$2/ppo_campaign_a8_final.zip"
}

# --- A: current shaping (baseline) ---
HIT_REWARD=1.0 MISS_PENALTY=0.05 MOVE_REWARD=0.002 LIVING_REWARD=-0.005 \
  run A ./.cache/ab/A

# --- B: combat-retuned shaping ---
HIT_REWARD=2.0 MISS_PENALTY=0.0 MOVE_REWARD=0.0003 LIVING_REWARD=-0.005 \
  run B ./.cache/ab/B

echo "================ A/B DONE ================"
