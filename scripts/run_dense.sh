#!/bin/zsh
# Diagnostic: can the pipeline learn to SHOOT when enemies are always present?
# Dense-combat scenario (defend_the_center), fresh, pro-combat shaping.
set -e
cd /Users/matheusfavaretto/Documents/labs/poc-doom-obisidyan
PY=.venv/bin/python
CK=./.cache/dense
rm -rf "$CK"; mkdir -p "$CK"

export CAMPAIGN=0 DOOM_SCENARIO=defend_the_center \
       DOCS_ENABLED=0 MEMORY_ENABLED=0 N_ENVS=4 SEED=42 \
       HIT_REWARD=2.0 MISS_PENALTY=0.0 MOVE_REWARD=0.0003 LIVING_REWARD=-0.005 \
       CHECKPOINT_DIR="$CK" RUN_NAME=dense-diag

echo "================ TRAIN dense (100k) ================"
$PY -m rl.train --fresh --timesteps 100000

echo "================ EVAL dense ================"
$PY -m rl.eval --episodes 10 --path "$CK/ppo_defend_the_center_a3_final.zip"
echo "================ DENSE DONE ================"
