<div align="center">

# HeLLMind

### A Self-Improving Reinforcement-Learning Agent for Doom,<br/>with Honest Measurement as a First-Class System

![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-638%20passing-brightgreen)
![local](https://img.shields.io/badge/100%25-local-orange)

*An empirical study, fully reproducible on a single laptop. No API keys, no cloud.*

</div>

---

## Abstract

We study whether a closed-loop supervisor can improve a reinforcement-learning agent
**unattended**, judged exclusively on honest evaluation metrics. The agent — a ~930k-
parameter CNN+MLP policy trained with PPO on ViZDoom (freedoom2) — perceives raw
pixels, a spatial-memory channel, a depth buffer and its own health/ammo, and acts
through 19 combined actions with **no gameplay assists**. The supervisor trains in
chunks, evaluates with tempered sampling, proposes one configuration change per
iteration informed by persistent memory (episodic, semantic, and cross-run behaviour
trends), keeps the change only if a composite score holds, and escalates structural
interventions when it plateaus.

**Findings.** (1) The closed loop works: in an unattended 8-iteration run it improved
its own composite score from −0.15 to 0.66 with six kept interventions, while death
rate fell 50%→0%. (2) Skills transfer: navigation trained on a small map lifted
exploration on an unseen map 2.6× (5.3%→13.9%) under a shared action space.
(3) Solo competence is learnable: 13.9 kills/episode at 30% shooting accuracy with no
aim assist (historic assisted ceiling: 3–5%). (4) Most importantly, *none of this was
measurable until the metrics stopped lying*: we document seven distinct measurement
failures — including a reward-hacking exploit the agent discovered in our geodesic
navigation gradient — each diagnosed from the system's own telemetry and sealed with a
regression test. The study's central claim: **in applied RL, the measurement system is
as much the artifact as the agent.**

---

## 1. Introduction and research questions

Reinforcement learning tutorials end where real projects begin: at the moment the
reward curve goes up but the agent is not actually doing what you wanted. This
repository is a longitudinal, fully-instrumented case study of that gap, organised
around three questions:

- **RQ1 — Closed-loop self-improvement.** Can a supervisor that proposes, validates
  and keeps/reverts its own configuration changes improve an agent *without a human in
  the loop*, when judged on deterministic/tempered evaluation rather than training
  reward?
- **RQ2 — Reward integrity.** What does it take for dense reward shaping to survive
  contact with an agent that probes it with tens of millions of trials?
- **RQ3 — Skill transfer.** Do navigation and combat skills learned on one map
  transfer to another when the brain family (observation space + action space) is held
  constant?

The contribution is not a new algorithm. It is an **engineering methodology** —
regime-local baselines, geodesic navigation metrics, metric contracts, exploit
forensics — plus the documented failure cases that motivated each piece.

## 2. Background: the RL concepts used here

*Readers familiar with RL can skip to §3.*

An **agent** (a neural network) interacts with an **environment** (Doom via ViZDoom).
Each step it receives an **observation**, emits an **action**, and collects a scalar
**reward**; learning adjusts the network so that high-return behaviour becomes more
probable. Nothing is labelled — the reward function *is* the specification, which is
precisely why it is the main attack surface (§5).

- **On-policy vs off-policy.** PPO (used here) learns only from data the current
  policy just generated — stable, but data-hungry. QR-DQN (alternative engine) replays
  past experience — sample-efficient, but here it repeatedly converged to degenerate
  camping behaviour.
- **Entropy and the argmax trap.** A PPO policy is a distribution over actions; its
  entropy measures indecision. When entropy collapses, the *single most probable*
  action can freeze into something degenerate while the distribution as a whole
  remains competent. Measured here: argmax 0 kills vs tempered-sampling 2.3 kills on
  the identical network — which is why all evaluation uses temperature 0.5.
- **Reward shaping.** The native game reward is too sparse to bootstrap from, so dense
  intermediate signals are added (exploration, combat, route progress). Shaping is
  expressed as a *potential-based, signed* term wherever possible so that round trips
  pay zero (§3.4).
- **Exploration.** Frontier/coverage bonuses (pay only new ground), RND curiosity, and
  Go-Explore return-goals: remember the deepest point ever reached and sometimes start
  an episode with a dense gradient back to it.

## 3. System and methods

### 3.1 Agent

| Component | Specification |
|---|---|
| Observation | 6×84×84 (grayscale ×2 frames, spatial-memory, depth ×2) + 4 game variables (health, ammo…) |
| Policy | CNN + MLP fusion (`MultiInputPolicy`), ≈930k parameters |
| Actions | 19 discrete combined actions (move+turn+shoot, strafe+shoot, retreat+shoot, USE, weapon switch) |
| Trainer | PPO (SB3), LR 2.5e-4 with linear decay **floored at 10%** (see §5, "frozen brain") |
| Assists | **None** during solo experiments (`--no-assists`); doors open on contact (`SOLO_AUTO_USE=1`) — map mechanics, not skill |

### 3.2 The supervisor (closed loop)

Per iteration: **train** 300k steps (resuming — brains are never discarded) →
**evaluate** N=10–20 episodes, temperature 0.5 → **score** on a selectable objective
profile → **propose** one configuration change → **keep/revert** against the
regime-local best → **escalate** on plateau (reset knobs → switch map *unless the run
is map-pinned* → revert to regime-best config + raise entropy → archive the config
history; the brain survives every level).

The proposal step consults, in priority order: per-eval metric diagnosis; cross-run
behaviour trends (a flag must persist in ≥60% of runs to act); a persistent memory
policy that never repeats a disproven change; semantic recall over past situations
(local embeddings); optionally a local LLM constrained to a validated parameter
registry.

**Regime-local baselines.** Scores are only ever compared within a *regime* — a
segment of iterations with the same map, config era and metric definitions. Escapes
and map switches start a new regime with a fresh baseline. This rule was paid for
three times (§5).

### 3.3 Honest navigation measurement: the geodesic field

Straight-line distance is meaningless in a maze. We rasterise the map's walls from the
WAD geometry into a 64-unit grid and BFS from the exit, yielding distance **along
walkable routes**, with three refinements that each fixed an observed failure:

1. **Directional steps** — falling any height is legal; climbing >24 units is not
   (Doom's step limit). An undirected model declared the exit unreachable.
2. **Sub-body slits** — the engine floors the player at the highest sector its
   16u-radius body touches, so pits narrower than 32u are walkable. A point model fell
   into a decorative 8u slit.
3. **Connectivity-checked sources + off-route penalty** — BFS sources must be cells a
   player could step to the exit from, and positions off the field read
   `max_field + euclidean` (strictly worse than anywhere on the route). Both rules
   exist because the agent *learned to jump into a pit* that an unconditional seeding
   had flooded with near-zero distances (§5, "the pit dive").

From this field derive: the shaping gradient, `route_progress` (1 − d(closest
reached)/d(spawn)), `death_route_dist` (where on the route deaths occur), and
route-aware Go-Explore goals (off-route cells can never be goals).

### 3.4 Measurement methodology

- **Tempered evaluation** (T=0.5), optional `--seed` pinning all RNGs (env + torch +
  numpy).
- **Per-episode distributions**: every evaluation reports mean, sample std (ddof=1),
  median, n, and a **95% Student-t confidence interval** for each per-episode metric.
- **Metric contract**: a tracker aggregate only "exists" when it reaches the
  `METRICS_JSON` consumer dict — enforced by test, because one metric was computed for
  a full evaluation while every consumer read `None`.
- **Methodology capture**: each evaluation embeds its own conditions (episodes,
  temperature, seed, assists state, brain checkpoint) into its metrics.
- **`doom-cli report`** renders the full study: trajectory with KEEP/REVERT and regime
  boundaries, metric trajectories, CI tables, the exact formulas, and a limitations
  section.

## 4. Results

All values from tempered evaluation with skill assists off, single seed unless noted.
Dates identify the measurement era (metric definitions changed during the study — see
threats to validity).

| Result | Value | Date |
|---|---|---|
| **RQ1** — unattended self-improvement (8 iters) | score **−0.15 → 0.66**, 6/8 kept; deaths 50%→0% | 2026-06-11 |
| **RQ2** — honest route penetration after all integrity fixes | best **47.7%** of the true 5,504u route | 2026-06-11 |
| **RQ3** — transfer: nav-map training → unseen MAP01 | exploration **5.3% → 13.9%** (2.6×) | 2026-06-10 |
| Solo combat (MAP02) | **13.9 kills/ep**, 30% accuracy (assisted-era ceiling: 3–5%) | 2026-06-10 |
| Exit rate, small nav map (transferable 19-action brain) | **90%** | 2026-06-10 |
| Exit rate, campaign maps | **0%** — open problem | — |
| Solo exploration record (MAP01) | 21% | 2026-06-11 |

The campaign exit remains open. Every *structural* blocker identified to date —
inactive gradient, mis-scaled gradient, euclidean metric, three doors locked by the
assist policy, a 120u cliff, an 8u slit, the pit exploit, episode clock, plateau map
rotation — has been eliminated and regression-tested; a scripted probe confirms the
agent crosses the remaining bottleneck door in ~50% of approach attempts. What remains
is behavioural consolidation, currently addressed by route-aware Go-Explore.

## 5. Case studies: seven measurement failures (and their general lessons)

Each was discovered from the system's own telemetry, in production, after tests passed.

| # | Case | General lesson |
|---|---|---|
| 1 | **Assists confound.** Training metrics looked strong while assists did the aiming; solo, the network collapsed. | If a helper acts during training, the learner is graded on someone else's work. |
| 2 | **Frozen brain.** Global LR-schedule progress ≈ 1 on a mature brain → every resumed chunk trained at ~0% LR for weeks. | Monitor the *effective* learning rate; `approx_kl≈0` + `clip_fraction=0` = nothing is learning. |
| 3 | **Argmax collapse.** Deterministic policy froze to "stand still"; sampled policy fought fine. | Evaluate argmax *and* sampled; a large gap is an entropy problem, not a skill problem. |
| 4 | **Euclidean lie.** Exit 600u away in a straight line — through a wall; real route 5,504u. The gradient pinned the agent against that wall for three runs. | In maze-like state spaces, straight-line distance is not progress. |
| 5 | **Three locked doors.** Solo mode disabled USE; the route has three doors; failure was indistinguishable from "bad exploration" — except the metric ceiling was suspiciously exact. | A *pinned, exact* ceiling across runs indicates a hard constraint, not a soft skill gap. |
| 6 | **The pit dive** (reward hacking, the central case). Mis-seeded BFS + euclidean fallback made an inescapable pit read as "93% route progress"; the agent learned to dive in and idle. The tell: the pinned value was not a multiple of the grid size. | The agent searches the reward function with millions of trials — far more thoroughly than review. Every dense reward needs an explicit exploit hunt: *what is the cheapest state that scores well?* |
| 7 | **Metric that never shipped.** The tracker computed `route_progress` for a full evaluation while every consumer read `None` (inline dict never extended). | A metric exists only at the consumer. Contract-test the boundary. |

## 6. Threats to validity / limitations

- **Single-seed trajectories.** Per-evaluation CIs capture episode variance, not
  run-to-run variance; cross-configuration claims require multi-seed replication
  (supported via `doom-cli experiment` and `eval --seed`, not yet applied to the
  headline numbers).
- **Metric eras.** `route_progress` changed definition during the study; the report
  generator marks regime boundaries and never aggregates across them, but historical
  comparisons before 2026-06-11 are not commensurable.
- **Small n.** 10–20 episodes per evaluation; the loop's keep/revert margin (0.05) is
  a heuristic, not a significance test.
- **Single environment family.** One game, one brain family; external validity beyond
  ViZDoom is argued by analogy, not measured.
- **Compute bound.** All experiments on one Apple M-series laptop (~300–900 env
  steps/s); per-hypothesis cost is hours, shaping which hypotheses were affordable.

## 7. Reproducibility

```bash
git clone https://github.com/MatheuslFavaretto/HeLLMind.git && cd HeLLMind
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# the headline experiments
doom-cli auto --no-assists --map MAP02            # solo combat (RQ1 machinery)
doom-cli curriculum2 --stages mywh,navigate        # transfer pipeline (RQ3)
SOLO_AUTO_USE=1 SCORE_PROFILE=exit doom-cli auto --no-assists --map MAP01 --goexplore  # exit hunt (RQ2)

# honest measurement
doom-cli eval --temperature 0.5 --seed 42 --json   # CI-bearing, seeded evaluation
doom-cli report                                    # the full study: charts, CIs, formulas
doom-cli shell                                     # interactive ('/' opens the palette)
```

Optional: [Ollama](https://ollama.com) with `qwen2.5:3b` (batch documentation) and
`nomic-embed-text` (semantic memory). Everything degrades gracefully without them.
Artifacts: the run trail (`vault/.memory/autonomy.jsonl`) is the raw data behind every
chart; `scripts/backup_vault.sh` snapshots memory + brains + config.

## 8. Repository structure

```
doom/             environments (campaign.py), WAD geometry & geodesic field (geodesic.py)
rl/               train · eval (metric contract) · autonomous (the supervisor) · curriculum
writer/           memory stores · semantic memory · behaviour trends · thesis_report
instrumentation/  per-episode statistics, CIs, metrics contract
scripts/          verify_exit_route (scripted route prover) · record_demo · backup_vault
tests/            638 tests — every production failure in §5 has a regression test
vault/            generated knowledge base (Obsidian notes, brains, memory)
```

```bash
python -m pytest tests/ -q     # 638 tests, ~15s
```

## Citing

```bibtex
@misc{hellmind2026,
  title  = {HeLLMind: A Self-Improving RL Agent for Doom with Honest
            Measurement as a First-Class System},
  author = {Favaretto, Matheus},
  year   = {2026},
  url    = {https://github.com/MatheuslFavaretto/HeLLMind}
}
```

## License

MIT
