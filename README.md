<div align="center">

# HeLLMind

**Hell (Doom) · LLM · Mind** — a self-improving reinforcement-learning agent for Doom
that documents its own training into an Obsidian knowledge graph.

![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-600%20passing-brightgreen)
![local](https://img.shields.io/badge/100%25-local-orange)

*Runs entirely on local hardware. No API keys, no cloud dependency.*

</div>

---

## Overview

HeLLMind trains a neural network to play Doom (ViZDoom + freedoom2) with PPO or QR-DQN.
What distinguishes it from a standard RL training script is the closed loop built around
the agent:

- **Rich perception** — beyond pixels: spatial memory, depth buffer, health/ammo state,
  ground-truth enemy detection, optional automap and semantic channels.
- **Persistent memory** — deaths, frontiers, exits, and validated configuration changes
  survive across runs (JSONL source of truth, SQLite query view, vector DB for semantic
  recall) and feed back into training decisions.
- **A supervising loop** (`doom-cli auto`) — trains in chunks, evaluates honestly,
  proposes one configuration change per iteration, keeps it only if the composite score
  holds, and reverts it otherwise. Structural interventions (map rotation, config resets)
  fire automatically when the loop plateaus.
- **Self-documentation** — a local LLM (Ollama) distils events into lessons, hypotheses,
  and an Obsidian graph, in batch, without ever blocking the RL loop.
- **Human-in-the-loop options** — record your own play and bootstrap the policy via
  behavioral cloning.

## Key results (measured, dated)

All numbers from deterministic or tempered (T=0.5) evaluation. *Solo* = all gameplay
assists disabled (`--no-assists`): the network aims and navigates by itself.

| Result | Value | Date |
|---|---|---|
| Solo kills per episode (MAP02) | **13.9** (baseline ~1) | 2026-06-10 |
| Solo shooting accuracy | **30%** (historic ceiling 3–5%) | 2026-06-10 |
| Skill transfer, nav → MAP01 | exploration **5.3% → 13.9% (2.6×)** | 2026-06-10 |
| Exit rate, my_way_home (transferable 19-action brain) | **25%** | 2026-06-10 |
| Exit rate, my_way_home (dedicated 5-action brain) | 93% at 901k steps | 2026-06-05 |
| Exit rate, campaign maps | 0% — open problem | — |

Two caveats apply to numbers older than 2026-06-10. First, all long-trained brains
learned with gameplay assists enabled, which corrupts credit assignment (the network is
rewarded for actions the assist executed); solo performance is the honest measure.
Second, a learning-rate scheduling bug silently froze every resumed training chunk
(effective LR ≈ 0%); it is fixed (`LR_MIN_FACTOR`), and older numbers understate the
architecture's capability.

## Architecture

```
                      ┌──────────────────── THE AGENT ────────────────────┐
 ViZDoom (Doom) ────▶ │  SENSES                    BRAIN                  │ ──▶ action
                      │  pixels · spatial memory   CNN + MLP fusion       │
                      │  depth · health/ammo       (MultiInputPolicy)     │
                      │  enemy detection           PPO or QR-DQN          │
                      └─────────────────────┬──────────────────────────────┘
                                            │ every episode
                                            ▼
 ┌────────────────────── MEMORY (persists across runs) ──────────────────────┐
 │ deaths + context · frontier cells · exit positions · lessons · learned    │
 │ config · semantic vector DB                                               │
 └─────────────────────────────────┬──────────────────────────────────────────┘
                                   │ informs decisions
                                   ▼
 ┌────────────────────── COACH (the self-improvement loop) ──────────────────┐
 │ behaviour flags → hypotheses → A/B experiments → adopt what is proven     │
 │ reward auto-tuning · plateau escape · curriculum · batch LLM documentation │
 └────────────────────────────────────────────────────────────────────────────┘
```

### Perception

| Channel | Purpose | Default |
|---|---|---|
| Pixels (84×84, frame-stacked) | primary view | on |
| Spatial memory | second channel marking visited areas | on |
| Depth buffer | per-pixel distance — 3D structure for navigation | on |
| Health + ammo | the agent knows its own state (DFP/Arnold approach) | on |
| Enemy detection (labels) | ground-truth "enemy on screen" | on |
| Semantic channel | detections painted into the input by category | off (`SEMANTIC_CHANNEL=1`) |
| Automap | top-down explored layout | off (`AUTOMAP=1`, ~10% throughput cost) |

### Action space

15 combined actions (movement + turning + attack + use + weapon switch), or 19 with
`STRAFE=1` (adds strafe-while-firing and retreat-while-firing). The brain checkpoint
name encodes every flag that changes the observation shape or action count, so
incompatible brains can never cross-load.

### Training engines

- **PPO** (default) — stochastic policy. Evaluation uses tempered sampling
  (`--temperature 0.5`) rather than pure argmax: a low-entropy policy's argmax can
  collapse onto a single passive action while the learned distribution is sound.
- **QR-DQN** (`--algo dqn`) — off-policy with replay buffer; more sample-efficient on
  discrete actions. The supervising loop is engine-aware (tunes `DQN_EPS_FINAL` instead
  of `ENT_COEF`).

### Reward design

Four active signal groups (a deliberate reduction from ~12 — surplus shaping terms are
reward-hacking surface):

| Signal | Role |
|---|---|
| Combat (kill / hit / miss / death / damage) | primary objective |
| RND curiosity | rewards unfamiliar states; never saturates |
| Frontier + coverage | pays only net outward progress; circling cannot farm it |
| Exit proximity | dense gradient toward a known exit position |

Combat and exploration are decoupled at the reward level: enemy visible → combat focus;
screen clear → exploration focus. The loop measures the two regimes independently
(`combat_engagement`, `explored_fraction`) and tunes them with separate levers.

### The supervising loop

Each iteration: train a chunk (resuming the existing brain) → evaluate → score against
a selectable objective profile → propose one change → keep or revert. The proposal step
consults, in order: metric diagnosis (aim offset, wasted shots, revisit rate, reward
breakdown), cross-run behavior trends, the persistent memory policy (death patterns;
never repeats a disproven change), semantic recall of similar past situations, and
optionally an LLM with the full parameter registry.

**Objective profiles** (`SCORE_PROFILE`): `combat` (default — aim quality first) or
`exit` (completion and survival first). The same loop optimises different goals on
different maps without code changes.

**Plateau escape.** When no improvement holds for N iterations, the loop escalates
structural interventions instead of further reward nudges: reset knobs → switch map →
revert to the regime's best config and raise entropy → clear the reward-evolution
history (timestamped backup). The brain checkpoint is never discarded at any level, and
baselines are regime-local — scores from different maps or config eras are never
compared against each other.

## Installation

```bash
git clone https://github.com/MatheuslFavaretto/HeLLMind.git && cd HeLLMind
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional: [Ollama](https://ollama.com) with `qwen2.5:7b-instruct` for LLM documentation
and proposals. Everything degrades gracefully without it.

## Usage

```bash
# Train (the supervising loop is the recommended mode; it resumes by default)
doom-cli auto                          # train → eval → self-tune → repeat
doom-cli auto --no-assists             # solo mode: the network aims and navigates itself
doom-cli auto --map MAP02 --algo dqn   # specific map, QR-DQN engine
doom-cli curriculum2                   # transfer pipeline: nav (my_way_home) → MAP01 → full

# Watch and measure
doom-cli watch --overlay               # live window with HUD + minimap
doom-cli eval --temperature 0.5        # honest metrics: kills, exploration, exit rate
doom-cli benchmark                     # ablation: prove each layer adds value
doom-cli progress                      # learning curve across checkpoints
doom-cli intel                         # network architecture, parameters, memory, disk

# Cognition and memory
doom-cli diagnose                      # eval + behavior flags + next-step suggestion
doom-cli behavior --trends             # chronic behavior patterns across runs
doom-cli hypothesize                   # behavior flags → falsifiable hypotheses
doom-cli experiment                    # multi-seed A/B validation of a hypothesis
doom-cli recall / semantic / bestiary  # episodic, semantic, and monster memory

# Maintenance
doom-cli prune --apply                 # GC step-checkpoints (keeps _final + newest 10)
doom-cli status                        # brain + memory + config at a glance
```

Behavioral cloning from your own play:

```bash
python scripts/record_demo.py --map MAP01 --episodes 3 --strafe   # you play
doom-cli bc --epochs 10                                           # clone
doom-cli auto --map MAP01                                         # RL refines
```

## Configuration

All settings live in `.env`. Notable defaults:

| Setting | Default | Effect |
|---|---|---|
| `N_ENVS` | 8 | parallel environments (ViZDoom is CPU-bound) |
| `STRAFE` | 1 | 19-action space with dodge/retreat combos |
| `LR_MIN_FACTOR` | 0.1 | floors the LR schedule — resumed chunks always keep learning |
| `AUTO_PRUNE_KEEP` | 10 | in-loop checkpoint GC for the family being trained |
| `SCORE_PROFILE` | combat | what the loop optimises (`combat` / `exit`) |
| `AUTO_AIM` etc. | 1 | gameplay assists — disable with `--no-assists` for solo training |
| `SCENARIO_WAD` | — | PWAD overlay: scenario maps with the full campaign action space |

Reliability guarantees baked into the loop:

- **Resume is the default everywhere.** Brain weights are the only asset that compounds;
  no command discards a trained brain without an explicit `--fresh`.
- **`LR_MIN_FACTOR`** — SB3 computes schedule progress globally on resume, which used to
  freeze long-trained brains (measured: effective LR 9e-08, zero policy movement). Fixed
  and floored.
- **Train/eval parity** — curriculum stages build their environment once and share it
  between training and evaluation, so they can never silently run on different maps.
- **JSONL writes, SQLite reads** — documentation can never corrupt the source of truth.
- **Knowledge is adopted only if proven** — a config change enters `learned_config` only
  after surviving tempered evaluation.

## Project structure

```
doom/             ViZDoom environments (campaign.py is the main one), WAD parsing, RND, overlay
rl/               train (PPO) · train_dqn (QR-DQN) · eval · autonomous (the loop) · coach_graph
                  progressive_curriculum (transfer pipeline) · checkpoint_gc · bc · experiment
writer/           memory stores (episodic/coverage/exit/frontier) · db (SQLite) · semantic_memory
                  behavior · hypothesize · reflect · learned_config · LLM documentation
instrumentation/  metrics, stats tracking, Prometheus export
scripts/          record_demo · benchmark_device · probe_map · make_gif
tests/            600 tests, pytest
vault/            the Obsidian knowledge base (notes, brains, memory) — generated
```

## Testing

```bash
python -m pytest tests/ -q
```

600 tests cover the environments, training loops, the supervisor (scoring, plateau
escape, checkpoint GC), curriculum parity, memory stores, and the CLI. The project's
working rule: a feature is done when it has been observed working in a real run, not
when its tests pass — several production bugs were found by reading run telemetry that
tests had missed.

## License

MIT
