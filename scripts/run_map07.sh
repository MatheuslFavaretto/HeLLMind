#!/bin/zsh
# Campaign-mode combat bootstrap: dense-enemy arena map (MAP07), 8 actions.
# If this learns to kill, the brain transfers to MAP01-05 (same action space).
set -e
cd /Users/matheusfavaretto/Documents/labs/poc-doom-obisidyan
PY=.venv/bin/python
CK=./.cache/map07
rm -rf "$CK"; mkdir -p "$CK"

export CAMPAIGN=1 MAPS=MAP07 DOCS_ENABLED=0 MEMORY_ENABLED=0 N_ENVS=4 SEED=42 \
       HIT_REWARD=2.0 MISS_PENALTY=0.0 MOVE_REWARD=0.0003 LIVING_REWARD=-0.005 \
       KILLS_TO_CLEAR=5 CHECKPOINT_DIR="$CK" RUN_NAME=map07-bootstrap

echo "================ TRAIN MAP07 (200k, campaign) ================"
$PY -m rl.train --fresh --maps MAP07 --n-envs 4 --timesteps 200000

echo "================ EVAL MAP07 ================"
$PY -m rl.eval --episodes 10 --path "$CK/ppo_campaign_a8_final.zip"
echo "================ MAP07 DONE ================"
