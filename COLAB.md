# 🚀 Running HeLLMind on Google Colab (free GPU)

The honest reason to do this: **compute**. The ViZDoom champions trained on GPU clusters
for days. A free Colab GPU + Google Drive persistence + the `--resume` loop lets you
accumulate *far* more training frames than a laptop — without tying up your machine.

> **What works on Colab:** training, `auto`, `bc` (cloning), `eval`. **What does NOT:**
> recording human demos (`record_demo.py` needs a game window — Colab is headless). Record
> demos on your Mac, then train on Colab. The LLM docs (Ollama) are also local-only, so
> train with `--no-docs`.

---

## 0. One-time: push your local work to GitHub

The session's features (game_vars, BC, fixes) must be on GitHub before Colab can clone them.
On your Mac:

```bash
git add -A
git commit -m "Game-vars, BC, resume + auto fixes"
git push
```

---

## 1. Colab notebook — paste these cells

### Cell 1 — GPU check + system deps for ViZDoom
```python
!nvidia-smi -L   # confirm a GPU is attached (Runtime → Change runtime type → GPU)

# ViZDoom's native build deps (Linux)
!apt-get -qq install -y build-essential cmake libboost-all-dev libsdl2-dev \
    libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev \
    libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev \
    timidity libwildmidi-dev unzip > /dev/null
```

### Cell 2 — clone the repo + install Python deps
```python
# The full-agent code is on the feat/new-features branch (merge it to main to drop -b).
!git clone -b feat/new-features https://github.com/MatheuslFavaretto/HeLLMind.git
%cd HeLLMind
!pip -q install -r requirements.txt
```

### Cell 3 — mount Google Drive (so the brain/memory SURVIVE disconnects)
```python
from google.colab import drive
drive.mount('/content/drive')

import os
VAULT = '/content/drive/MyDrive/hellmind-vault'   # persists across Colab sessions
os.makedirs(VAULT, exist_ok=True)
os.environ['VAULT_PATH'] = VAULT
print('vault ->', VAULT)
```

### Cell 4 — (optional) upload your recorded demos
If you recorded demos on your Mac and want to bootstrap with BC, upload them to the Drive
vault at `MyDrive/hellmind-vault/.memory/demos/` (or use the Colab file uploader), then:
```python
!VAULT_PATH=$VAULT python -m rl.bc --epochs 10   # clone your human play
```

### Cell 5 — TRAIN with the GPU (resume-safe, no docs)
```python
# Verify torch sees the GPU
import torch; print('CUDA:', torch.cuda.is_available())

# Long auto run. --resume continues across Colab sessions (brain + history on Drive).
!VAULT_PATH=$VAULT python -m rl.autonomous --map MAP01 --iterations 20 --steps 100000
```

### Cell 6 — check progress any time
```python
!VAULT_PATH=$VAULT python -m rl.eval --episodes 20 --json --temperature 0.5
```

---

## 2. Surviving Colab disconnects (the key to "leave it running")

Free Colab drops the session after ~12 h or on idle. Because the brain, memory, and the
`autonomy.jsonl` trail live on **Google Drive**, you just reconnect and re-run **Cell 5** —
the `--resume` default picks up exactly where it left off. Run it day after day; the
ratchets (frontier archive, exit memory, learned config) keep accumulating.

## 3. Force the GPU (if SB3 stays on CPU)

SB3 auto-selects CUDA when available, so on a Colab GPU runtime it should "just work". If
not, the device is set inside `rl/train.py` (PPO `device="auto"`); CUDA will be picked
automatically on Colab.

## 4. Honest expectations

- The neural net is small and ViZDoom stepping is CPU-bound, so the GPU speedup is **modest
  per-step** — the real win is **wall-clock you don't pay for**: many more total frames
  accumulated over days. That total-frame count is the actual gap vs the champions.
- Everything we built (game_vars, RND, Go-Explore, BC, the cognitive loop) finally gets the
  training budget it needs to pay off here.
