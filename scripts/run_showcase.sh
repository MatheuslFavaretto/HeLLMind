#!/bin/zsh
# Showcase run: the autonomy goal in action — explore + complete + combat, WITH the
# agent's own spatial memory as a 2nd observation channel. Documented into the vault.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

export CAMPAIGN=1 MAPS=MAP02 DOCS_ENABLED=1 MEMORY_ENABLED=1 N_ENVS=4 SEED=42 \
       SPATIAL_MEMORY=1 COVERAGE_REWARD=0.5 EXIT_REWARD=200.0 EPISODE_TIMEOUT=4200 \
       HIT_REWARD=2.0 MISS_PENALTY=0.0 MOVE_REWARD=0.0 LIVING_REWARD=-0.005 \
       DEATH_PENALTY=5.0 KILLS_TO_CLEAR=5 WRITE_EVERY_STEPS=50000 \
       OLLAMA_MODEL=qwen2.5:7b RUN_NAME=map02-explore
# Into the vault (no CHECKPOINT_DIR override). --fresh: obs space changed (2 channels).

echo "================ SHOWCASE TRAIN+DOCS (MAP02, explore+memory, 500k) ================"
$PY -m rl.train --fresh --maps MAP02 --n-envs 4 --timesteps 500000

echo "================ EVAL ================"
$PY -m rl.eval --episodes 12
echo "================ SHOWCASE DONE ================"
