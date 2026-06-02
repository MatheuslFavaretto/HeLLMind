<div align="center">

# 🔥 HeLLMind

**He·LLM·ind** — *Hell* (Doom) + *LLM* + *Mind*

A **Reinforcement Learning** agent plays Doom while a **local LLM** documents its own
learning into an **Obsidian knowledge graph**. 100% local, no cost.

![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-61%20passing-brightgreen)
![local](https://img.shields.io/badge/100%25-local-orange)

</div>

| 🎮 Agent playing | 🗺️ Real map + path | 📈 Performance (5%→18% accuracy) |
|:---:|:---:|:---:|
| ![](assets/doom-frame.png) | ![](assets/minimap-real.png) | ![](assets/performance.png) |

> Training and the LLM are **decoupled**: Ollama never runs inside the PPO loop (that
> froze training) — notes are generated in batch, at the end. It even works **without
> Ollama** (factual notes) and trains **in batches** (the brain is reused by default).

---

## 🎯 The problem & why it matters

Training an RL agent produces an **ocean of opaque numbers**. You watch `ep_rew_mean`
tick up and down with almost no insight into *what the agent actually learned*, *why it
fails*, or *how this run compares to the last one* — and nothing is remembered across
runs. The obvious fix (have an LLM narrate the training) **breaks the training**: calling
a model inside the RL loop stalls every environment for seconds.

**HeLLMind** turns a training run into a **navigable, self-documenting knowledge graph**
with **memory that persists across runs** — and does it *without ever blocking the PPO
loop*, **100% locally** (no API key, no cost). The payoff:

- **Understand** the agent: behavior changes, aim, the path it walked (on the real map),
  regressions, and the learning arc — in prose, not just plots.
- **Remember & learn across runs**: a persistent event memory feeds reusable *lessons*
  and even **reweights the next curriculum** toward the maps the agent fails on.
- **Steer** it: edit one Obsidian note to change training live; get reward-tweak
  suggestions grounded in observed behavior.

## 🏗️ Architecture

Three layers linked by files on disk — heavy work never touches the PPO loop (±2% FPS
budget). Two feedback loops close back onto training.

```
  edit control.md ─────────────┐ (live steering: stop / cadence / novelty)
                               ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ 1. TRAINING  (real-time, never blocks)                                  │
 │    ViZDoom → PPO CnnPolicy (N envs) ─ reward: +hit −miss −damage −death │
 │    • CheckpointCallback → ./<vault>/.checkpoints/*.zip   (the "brain")  │
 │    • DocCallback → snapshots .cache/pending_runs/*.jsonl (fast, no LLM) │
 │    • MemoryRecorder → <vault>/.memory/episodic/*.jsonl   (death/success)│
 └───────────────────────────────┬───────────────────────────────────────┘
                                  │  end of training
                                  ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ 2. POST-PROCESSING  (batch — the ONLY place the LLM runs)               │
 │    writer.process_run: checkpoint notes · concepts · real minimap ·    │
 │    synthesis · regression · run comparison · lessons · reward suggest   │
 └───────────────────────────────┬───────────────────────────────────────┘
                                  ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ 3. OBSIDIAN VAULT  (knowledge graph + persistent memory)               │
 │    10-checkpoints · 20-concepts · 30-runs · 40-maps · 50-compare ·      │
 │    60-lessons · 00-index/Knowledge Graph.md (MOC hub)                   │
 └───────────────────────────────┬───────────────────────────────────────┘
                                  │  memory of deaths per map
                                  └────►  reweights the next CURRICULUM
                                         (agent trains more where it fails)
```

> Works **without Ollama** (notes fall back to factual mode) and **trains in batches**
> (the brain is tied to the vault and reused by default).

## 🗂️ How the vault assembles itself

The agent writes `.md` straight into the Obsidian folder; the **Graph View** forms itself.

```
vault/
├── 10-checkpoints/   CKPT-0003-step7500.md       ← what changed + minimap + evidence
├── 20-concepts/      Concept - Policy Entropy.md  ← reusable RL concepts (stable id)
├── 30-runs/          run-demo10k.md + Synthesis   ← index + the run's "story"
├── 40-maps/          Map - MAP01.md               ← per-map progress (campaign)
├── 50-compare/       Comparison - A-vs-B.md       ← run comparison
├── 60-lessons/       Lessons.md                   ← cross-run lessons (cognitive memory)
├── attachments/      *.png                        ← minimaps and curves
└── 00-index/         Knowledge Graph.md · control.md · Reward Suggestions.md
```

<details>
<summary>📄 Example checkpoint note (auto-generated)</summary>

```markdown
---
type: checkpoint
timesteps: 7500
shooting_accuracy: 0.18
regression: true
map: MAP01
---
# Aim improving, but taking too much damage

> [!warning] Regression detected
> Possible forgetting — see [[Concept - Catastrophic Forgetting]]
> - mean reward dropped from 79.8 to 50.8 (-36%)

## What changed in behavior
Accuracy rose from 15% to 18%, indicating better aim...

## Level minimap   ![[CKPT-0003-step7500.png]]
## RL concepts
- [[Concept - Exploration vs Exploitation]]
- [[Concept - Policy Entropy]]
```
</details>

## 🚀 Getting started

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # set VAULT_PATH

# (optional, for narrated notes) local Ollama:
brew install ollama && ollama serve && ollama pull qwen2.5:3b

python3 -m rl.train                              # train + document at the end
python3 -m rl.train --campaign --maps MAP01      # full maps (shows the path)
python3 -m rl.train --no-docs --render           # just play, with a window
python3 -m rl.status                             # saved checkpoints + progress
```

> **The brain is tied to the vault.** Running again with the same `VAULT_PATH`
> **continues where it left off** (doesn't restart); another vault starts from zero.
> Use `--fresh` to reset on purpose. Better notes without retraining:
> `python3 -m writer.process_run --model qwen2.5:7b`

## ✨ Features

- **Never freezes** — LLM decoupled, runs in batch post-training.
- **Rich signal** — aim (hits/misses), damage, **path & coverage**, weapons, entropy.
- **Real minimap** — actual level walls (ViZDoom sectors) + path heatmap.
- **Connected graph** — a `Knowledge Graph` hub (MOC) links runs, maps, concepts and
  lessons; concepts use **deterministic IDs** (no broken/duplicate links).
- **Interprets, not just describes** — detects **regression** and links *Catastrophic Forgetting*.
- **Run synthesis** — the LLM tells the learning arc in a single note.
- **Persistent memory** — episode events are stored across runs; an offline LLM pass
  extracts reusable **lessons** (e.g. "the agent dies in corridors below 30 HP").
- **Compares runs** — table + charts + verdict (`writer.compare_runs`).
- **Closed loop (memory → training)** — the cross-run memory of *where the agent died*
  reweights the campaign curriculum, so it automatically **trains more on the maps it
  fails on**. Cognition stops only informing the human and starts improving the agent.
- **Reward suggestions** — an offline LLM proposes reward-weight tweaks from observed
  behavior (e.g. "raise damage penalty: 92% of deaths at low HP"); you approve via `.env`.
- **Obsidian → training** — edit `control.md` and training adapts without restarting.
- **Continuous learning** — the brain lives in the vault and is **reused automatically** (`--fresh` resets) · **works without Ollama** · **61 tests** (`pytest -q`).

## 📊 Evaluate & prove performance

Training metrics are noisy (exploration + shaping). To measure what a brain *actually*
learned, evaluate it **deterministically**:

```bash
python3 -m rl.eval --episodes 50      # clean: mean reward, accuracy, kills/ep, success
```
> Real example: a 150k `defend_the_center` brain reads **25% accuracy** during training
> but **48% / 3.0 kills per episode** under deterministic eval — exploration was hiding it.

**Prove a feature helps (A/B).** Change one thing on the *same* task and compare:

```bash
# A: with the aim shaping (defaults)        B: baseline, shaping off
VAULT_PATH=./vault-A RUN_NAME=run-A python3 -m rl.train --timesteps 200000 --fresh
VAULT_PATH=./vault-B RUN_NAME=run-B HIT_REWARD=0 MISS_PENALTY=0 \
  DAMAGE_TAKEN_PENALTY=0 DEATH_PENALTY=0 python3 -m rl.train --timesteps 200000 --fresh
# Judge on shaping-independent numbers (RAW reward / accuracy / kills), not shaped reward:
VAULT_PATH=./vault-A python3 -m rl.eval --episodes 50
VAULT_PATH=./vault-B python3 -m rl.eval --episodes 50
```
For rigor, run a few **seeds** per side (`SEED=...`) and compare the *means* — RL is
noisy, and a single seed can mislead. Live curves: `tensorboard --logdir tb`.

## 📈 Expected evolution

![evolution](assets/evolution.png)

A classic curve (fast early gains → saturation) in 3 phases: **exploration →
convergence → refinement**. Each run's real curve lands in `30-runs/<run>.md`.

## 📁 Layout

```
doom/        env + campaign + map geometry         instrumentation/ metrics & tracker
rl/          training, eval, callbacks, curriculum, control, status, memory recorder
writer/      LLM, notes, minimap, charts, analysis, compare, process_run, memory, reflect, suggest
```

## 🧪 Tests

```bash
python3 -m pytest -q     # 61 tests, no ViZDoom/Ollama needed (synthetic info)
```

## 📜 License

MIT — see [LICENSE](LICENSE). Backlog and next steps in [TODO.md](TODO.md).
