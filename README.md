<div align="center">

# HeLLMind

**Hell (Doom) · LLM · Mind** — a self-improving reinforcement-learning agent for Doom
that documents its own training into an Obsidian knowledge graph.

![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-635%20passing-brightgreen)
![local](https://img.shields.io/badge/100%25-local-orange)

*Runs entirely on local hardware. No API keys, no cloud dependency.*

</div>

---

## Why this project is useful

Most RL repositories show you a training script and a reward curve. This one shows you
**what it actually takes to make reinforcement learning work** — because every classic
failure mode of RL happened here, was caught by the system's own instrumentation, and
was fixed with a regression test. If you want to learn RL beyond the tutorial level,
the value is in the [war stories](#war-stories--what-rl-actually-looks-like): real
reward hacking, real metric lies, real exploration traps, with the diagnosis trail
intact.

Beyond education, the project demonstrates three transferable patterns:

1. **A self-improvement loop that provably works.** The supervisor trains in chunks,
   evaluates honestly, proposes one configuration change, keeps it only if a composite
   score holds, and escalates structural interventions when it plateaus. In an
   unattended 8-iteration run it improved its own score from −0.15 to 0.66 with six
   kept interventions — no human in the loop. The same architecture (measure → propose
   → validate → keep/revert → escalate) applies to any optimisation problem, not just
   Doom.

2. **Honest measurement as a first-class system.** Tempered evaluation, geodesic
   metrics, reward-breakdown telemetry, death forensics — the instrumentation caught
   bugs that code review missed, including an exploit the agent itself discovered.
   The discipline is the product: *a metric only exists when it reaches the consumer,
   and a feature is only done when it's been observed working in production.*

3. **Local-first AI engineering.** Everything — training, evaluation, the LLM that
   writes documentation, the vector database for semantic memory — runs on a laptop.
   No API costs, no cloud, full reproducibility.

---

## Reinforcement learning, explained through this project

If you already know RL, skip ahead. If not, every concept below maps to something
concrete you can run here.

### The core loop

An **agent** (a neural network) lives in an **environment** (Doom, via
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom)). At every step it receives an
**observation** (what it sees), picks an **action** (move, turn, shoot…), and receives
a **reward** (a number). RL is the process of adjusting the network so that actions
which lead to more total reward become more likely.

Nobody labels anything. The agent is never told "that's a door" or "shoot the monster".
It only ever learns from the reward number — which is why *designing* the reward is
where most RL projects quietly fail (see the war stories).

### What the agent senses

The observation here is a stack of 84×84 images plus a small vector:

| Channel | What it gives the agent |
|---|---|
| Pixels (grayscale, 2 frames stacked) | the raw view; stacking 2 frames lets it perceive motion |
| Spatial memory | a second image marking everywhere it has already been this episode |
| Depth buffer | per-pixel distance — explicit 3D structure for navigation |
| Health + ammo vector | it *knows* when it is weak (the approach of ViZDoom-competition winners) |
| Enemy detection | ground-truth "is an enemy on screen" from the engine's labels buffer |

A CNN reads the images, a small MLP reads the vector, and the two fuse into one policy
(`MultiInputPolicy`). The whole brain is ~930k parameters — small enough to train on a
laptop, large enough to fight at 30% shooting accuracy with no aim assist.

### Actions

19 discrete **combined** actions (forward+turn, strafe+shoot, retreat+shoot…). This
matters: an early version used 8 one-button actions and the agent literally could not
move and shoot at the same time — it learned to back into a wall and spray. The action
space *is* part of the hypothesis space.

### Reward shaping — the dangerous art

The base game reward (kills, level end) is far too sparse to learn from: a random
agent essentially never finishes a map, so it would never see a single positive
example. **Shaping** adds dense intermediate rewards:

| Signal | Pays for | Why it exists |
|---|---|---|
| Combat (kill/hit, miss/death penalties) | fighting well | the primary objective |
| Coverage + frontier | net *outward* exploration | circling can't farm it |
| RND curiosity | visiting unfamiliar states | never saturates |
| Exit proximity (geodesic) | each step *along the real route* toward the exit | the only dense signal that exists before the first exit ever happens |

Every shaping term is **attack surface for reward hacking** — the agent will find any
exploit in your reward faster than you will (it tried ~37 million experiences here;
you reviewed the code once). This project's pit-dive story below is a perfect specimen.

### The two training engines

- **PPO** (default) — *on-policy*: learns only from data the current policy just
  generated. Stable, well-understood. Its policy is a probability distribution over
  actions; it explores by sampling from it. The distribution's **entropy** measures
  how undecided it is — and entropy management matters (see "argmax collapse" below).
- **QR-DQN** (`--algo dqn`) — *off-policy*: keeps a replay buffer and reuses every
  experience many times. More sample-efficient on discrete actions, but here it
  repeatedly converged to camping in a corner. Kept as an alternative engine.

### Evaluation honesty (the part most projects skip)

The training curve **lies**. `ep_rew_mean` going up tells you the *stochastic* policy
is collecting shaped reward — it says nothing about whether the agent actually clears
rooms or reaches exits. This project measures with:

- **Tempered evaluation** (`--temperature 0.5`): act from a sharpened version of the
  learned distribution. Pure argmax can "collapse" — when entropy drops too low, the
  single most-probable action freezes into something degenerate (stand still) while
  the distribution as a whole is still good. Measured here: argmax 0 kills, tempered
  2.3 kills, same network.
- **Geodesic route metrics**: distance to the exit measured *along walkable paths*
  (BFS over the map's wall geometry), not in a straight line. Euclidean "progress"
  pointed the agent into a wall for three full training runs.
- **Forensics**: every death is recorded with position, nearest enemy, weapon, ammo —
  so "why does it die?" is a database query, not a guess.

### Exploration — the actual hard problem

Doom maps are mazes with combat. The agent's exploration stack:

- **Frontier/coverage rewards** — pay only for *new* ground.
- **RND** (Random Network Distillation) — an intrinsic "have I seen this before?" bonus.
- **Go-Explore** — remember the deepest cell ever reached (across all runs, on disk);
  at episode start, sometimes set it as a return goal with a dense gradient. This
  project's version is **route-aware**: goals are weighted by geodesic depth along the
  route to the exit, and cells off the walkable route (pits) can never become goals.

---

## War stories — what RL actually looks like

Every one of these happened in this repository, was diagnosed from the system's own
telemetry, and is now locked by a regression test. They are the curriculum.

| Story | The lesson |
|---|---|
| **The assists confound.** Gameplay assists (auto-aim, auto-door) made training metrics look great while the network stayed vestigial — remove the crutch and it couldn't play. All "progress" belonged to the assists. | If a helper acts during training, the learner is being graded on someone else's work. Measure *solo*. |
| **The frozen brain.** The LR schedule used global progress; every resumed training chunk on a mature brain ran at ~0% learning rate. Weeks of "training" barely updated weights. | Plot the *effective* learning rate. `approx_kl ≈ 0` + `clip_fraction = 0` means nothing is learning, regardless of FPS. |
| **The argmax collapse.** Low entropy froze the deterministic policy into "stand still" while the sampled policy fought fine. | Evaluate both argmax and sampled. A big gap = entropy problem, not skill problem. |
| **The euclidean lie.** The exit is 600 units away in a straight line — through a wall. The real route is 5,500 units. A euclidean gradient pinned the agent against that wall for three runs. | In any maze-like state space, straight-line distance is not progress. Measure along feasible paths. |
| **The three locked doors.** Solo mode disabled the USE action assist; the exit route passes through three doors; the agent had never learned to press USE. Every exit attempt was structurally impossible — and the failure looked exactly like "bad exploration". | When a metric ceiling is *suspiciously exact* across runs, suspect a hard constraint, not a soft skill gap. |
| **The pit dive** (the crown jewel). The geodesic field accidentally seeded distance-zero cells inside an inescapable pit next to the exit; the off-route fallback was euclidean. The agent *discovered* that jumping into the pit read as "93% route progress" and learned to dive in and sit there — 0 deaths, 100% timeouts, metric pinned. The tell: the pinned value wasn't a multiple of the grid size. | The agent searches your reward function with millions of trials. Every dense reward needs an exploit-hunt pass: *what is the cheapest state that scores well?* |
| **The self-improvement loop that finally worked.** After the metrics were honest: 8 unattended iterations, 6 kept interventions, score −0.15 → 0.66, exploration and survival records. | Closed-loop optimisation works — but only on top of measurements that don't lie. |

---

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
 │ config · semantic vector DB (local embeddings)                            │
 └─────────────────────────────────┬──────────────────────────────────────────┘
                                   │ informs decisions
                                   ▼
 ┌────────────────────── COACH (the self-improvement loop) ──────────────────┐
 │ behaviour flags → hypotheses → A/B experiments → adopt what is proven     │
 │ reward auto-tuning · plateau escape ladder · batch LLM documentation      │
 └────────────────────────────────────────────────────────────────────────────┘
```

### The supervising loop in detail

Each iteration of `doom-cli auto`:

1. **Train** a chunk (300k steps by default), resuming the existing brain. Brains are
   never discarded — weights are the only asset that compounds.
2. **Evaluate** with tempered sampling on the configured objective profile
   (`SCORE_PROFILE`: `combat` = aim quality first, `exit` = completion first).
3. **Propose** one configuration change, consulting in order: metric diagnosis
   (aim offset, wasted shots, revisit rate, reward breakdown), cross-run behaviour
   trends, the persistent memory policy (never repeats a disproven change), semantic
   recall of similar past situations, and optionally a local LLM with the full
   parameter registry.
4. **Keep or revert** against the regime-local best score.
5. **Escalate** when stuck (the plateau ladder): reset knobs → switch map (unless the
   run is pinned to a map) → revert to the regime's best config and raise entropy →
   archive the poisoned config history and restart the evolution. The brain survives
   every level.

Baselines are **regime-local**: scores from different maps, metric definitions, or
config eras are never compared against each other — a lesson paid for three times.

---

## Measured results (dated, single-seed unless noted)

*Solo* = all skill assists disabled: the network aims and navigates itself
(doors still open on contact — map mechanics, not skill).

| Result | Value | Date |
|---|---|---|
| Solo kills per episode (MAP02) | **13.9** (baseline ~1) | 2026-06-10 |
| Solo shooting accuracy | **30%** (historic ceiling 3–5%) | 2026-06-10 |
| Skill transfer, nav-map → MAP01 | exploration **5.3% → 13.9%** (2.6×) | 2026-06-10 |
| Exit rate, my_way_home (transferable 19-action brain) | **90%** | 2026-06-10 |
| Unattended self-improvement run | score **−0.15 → 0.66**, 6/8 kept | 2026-06-11 |
| True-route penetration (honest geodesic metric) | best **47.7%**, survival ~100% | 2026-06-11 |
| Exit rate, campaign maps | **0%** — the open problem | — |

The campaign exit remains open: every structural blocker found so far (gradient,
scale, euclidean metric, doors, cliff, slit, pit exploit, clock) has been eliminated;
what remains is consolidating the door-corridor behaviour the agent already exhibits
occasionally. The route-aware Go-Explore mechanism targets exactly that.

---

## Installation

```bash
git clone https://github.com/MatheuslFavaretto/HeLLMind.git && cd HeLLMind
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional: [Ollama](https://ollama.com) with `qwen2.5:3b` (LLM documentation/proposals)
and `nomic-embed-text` (semantic memory embeddings). Everything degrades gracefully
without them.

## Usage

```bash
# Train (the supervising loop; resumes by default, never discards a brain)
doom-cli auto                          # train → eval → self-tune → repeat
doom-cli auto --no-assists             # solo mode: the network aims and navigates itself
doom-cli auto --map MAP01 --goexplore  # directed run (pinned map; plateau never rotates it)
doom-cli curriculum2                   # transfer pipeline: nav map → MAP01 → full

# Watch and measure
doom-cli shell                         # Claude-style interactive shell ('/' opens the palette)
doom-cli watch --overlay               # live window with HUD + minimap
doom-cli eval --temperature 0.5 --seed 42   # honest, reproducible metrics
doom-cli benchmark                     # ablation: prove each layer adds value

# Understand
doom-cli diagnose                      # eval + behaviour flags + next-step suggestion
doom-cli behavior --trends             # chronic patterns across runs
doom-cli recall / semantic / bestiary  # episodic, semantic and monster memory
doom-cli intel                         # network architecture, parameters, disk

# Maintain
doom-cli prune --apply                 # GC old checkpoints (keeps _final + newest 10)
scripts/backup_vault.sh                # memory + newest brains + config → tar.gz
```

## Configuration highlights (`.env`)

| Setting | Default | Effect |
|---|---|---|
| `SCORE_PROFILE` | combat | what the loop optimises (`combat` / `exit`) |
| `LR_MIN_FACTOR` | 0.1 | LR floor — resumed chunks always keep learning |
| `AUTO_PRUNE_KEEP` | 10 | in-loop checkpoint GC for the trained family |
| `EXIT_GEODESIC` | 1 | route-aware exit gradient (euclidean opt-out) |
| `SOLO_AUTO_USE` | 0 | under `--no-assists`, keep doors opening on contact |
| `GOEXPLORE_GOAL_PROB` | 0.4 | fraction of episodes that start with a return-goal |

## Project structure

```
doom/             ViZDoom envs (campaign.py), WAD geometry (geodesic.py), RND, overlay
rl/               train · train_dqn · eval · autonomous (the loop) · coach_graph
                  progressive_curriculum · checkpoint_gc · bc · experiment
writer/           memory stores · SQLite view · semantic_memory · behavior · hypothesize
                  frontier_store (route-aware Go-Explore) · LLM documentation
instrumentation/  stats tracker, metrics contract, Prometheus export
scripts/          verify_exit_route (scripted route prover) · record_demo · backup_vault
tests/            635 tests — every production bug became a regression test
vault/            the Obsidian knowledge base (notes, brains, memory) — generated
```

## Testing

```bash
python -m pytest tests/ -q
```

The suite covers the environments, both trainers, the supervisor (scoring profiles,
plateau ladder, checkpoint GC), the geodesic field (directional steps, slit handling,
pit exclusion), curriculum parity, memory stores, the metrics contract
(tracker → METRICS_JSON), and a dry-run of all 24 CLI entry points.

Working rule, learned the hard way: **a feature is done when it has been observed
working in a real run** — five production bugs in one week were found by reading run
telemetry after every test passed.

## License

MIT
