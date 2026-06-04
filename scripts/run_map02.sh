#!/bin/zsh
# Campaign-mode combat bootstrap: dense-enemy arena map (MAP02), 8 actions.
# If this learns to kill, the brain transfers to MAP01-05 (same action space).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CK=./.cache/map02
rm -rf "$CK"; mkdir -p "$CK"

export CAMPAIGN=1 MAPS=MAP02 DOCS_ENABLED=0 MEMORY_ENABLED=0 N_ENVS=4 SEED=42 \
       HIT_REWARD=2.0 MISS_PENALTY=0.0 MOVE_REWARD=0.0003 LIVING_REWARD=-0.005 \
       DEATH_PENALTY=5.0 KILLS_TO_CLEAR=5 CHECKPOINT_DIR="$CK" RUN_NAME=map02-bootstrap

echo "================ TRAIN MAP02 (200k, campaign) ================"
$PY -m rl.train --fresh --maps MAP02 --n-envs 4 --timesteps 200000

echo "================ EVAL MAP02 ================"
$PY -m rl.eval --episodes 10 --path "$CK/ppo_campaign_a8_final.zip"
echo "================ MAP02 DONE ================"
