#!/bin/zsh
# THE documented combat run: MAP02 (dense), 500k, into the vault, with Obsidian notes
# + cognitive memory. Uses the fixed death_penalty and the timeout-safe Ollama client.
set -e
cd /Users/matheusfavaretto/Documents/labs/poc-doom-obisidyan
PY=.venv/bin/python

export CAMPAIGN=1 MAPS=MAP02 DOCS_ENABLED=1 MEMORY_ENABLED=1 N_ENVS=4 SEED=42 \
       HIT_REWARD=2.0 MISS_PENALTY=0.0 MOVE_REWARD=0.0003 LIVING_REWARD=-0.005 \
       DEATH_PENALTY=5.0 KILLS_TO_CLEAR=5 WRITE_EVERY_STEPS=50000 \
       OLLAMA_MODEL=qwen2.5:7b RUN_NAME=map02-documented
# No CHECKPOINT_DIR override -> trains into the vault (./vault/.checkpoints).

echo "================ TRAIN+DOCS MAP02 (500k, into vault) ================"
$PY -m rl.train --fresh --maps MAP02 --n-envs 4 --timesteps 500000

echo "================ EVAL MAP02 (vault brain) ================"
$PY -m rl.eval --episodes 10
echo "================ DOC RUN DONE ================"
