# 📊 Curriculum results — what the agent can and can't do (measured, 2026-06-05)

Honest, deterministic-eval results from the V2 progressive curriculum
(`my_way_home → deadly_corridor → MAP01`). Each stage trains its own brain (the action
spaces differ: 5 / 7 / 15 actions), so these measure each skill **in isolation**.

## The numbers

| Stage | Map | Steps | exit-rate | death | explored | accuracy | kills/ep | Read |
|---|---|---|---|---|---|---|---|---|
| **my_way_home** | tiny, no enemies | 400k | **50%** | 0% | 38% | — | 0 | navigation works |
| **my_way_home** | tiny, no enemies | 901k | **93%** | 0% | 33% | — | 0 | more compute → near-perfect |
| **deadly_corridor** | lethal corridor (skill 5) | 800k | 0% | 100% | 15% | **81%** | 1.23 | aims + advances, dies before exit |
| **navigate** | freedoom2 MAP01 (real) | 1.0M | 0% | 15% | **4%** | — | 1.8 | the compute wall |

## What's proven ✅

- **The agent completes maps.** my_way_home: **93% exit-rate** at 901k steps (0% deaths,
  ~103-step solutions). The project's first exit-rate > 0 — the roadmap's headline milestone.
- **More compute works.** Same agent, 401k → 901k steps: exit-rate **50% → 93%**. Validates
  the V2 thesis "it's compute, not features."
- **Combat works.** deadly_corridor: **81% shooting accuracy**, advances down the corridor
  killing enemies (native reward 356). QR-DQN on MAP01 earlier: 2 kills/ep, 0% deaths,
  0.97 combat-engagement.
- **Mechanics are non-issues.** AUTO_USE opens doors on contact (verified); freedoom2 MAP01
  has zero keycards (probed) — the exit isn't gated.
- **The cognitive loop runs end-to-end.** train → eval → diagnose → tune → keep/revert, with
  the eval finally scoring DQN brains correctly (10 bugs fixed to get here).

## The wall 🧱

**Full freedoom2 MAP01 is a much bigger navigation problem than the toy scenarios.** With 1M
steps and max-exploration rewards (combat zeroed), the agent explores only **4%** — it can't
find its way out of the spawn area on a large, enemy-harassed, complex layout. This is the
gap V2_SPEC predicted: *the remaining gap is compute, not features.* The ViZDoom champions
trained tens of millions of steps; 1M on a laptop M5 (~30 min) isn't enough.

## Why my_way_home 93% but MAP01 4%

| | my_way_home | MAP01 freedoom2 |
|---|---|---|
| Size | tiny (purpose-built) | large, real level |
| Enemies | none | DoomImp / ShotgunGuy harassing |
| Exit | close to spawn | far, behind a complex layout |
| Result | 93% exit | 4% explored |

## Future work (needs compute)

1. **Cloud/Colab training** (`COLAB.md`) — accumulate 10M+ steps on a free GPU + resume loop.
   The honest path through the compute wall.
2. **deadly_corridor** needs a skill-curriculum (start at doom_skill 1-3) + millions of steps —
   it's the hardest ViZDoom benchmark.
3. **Bigger intrinsic-curiosity budget** for the large-map exploration problem.

> Reproduce any stage: `doom-cli curriculum2 --stages mywh` (or `corridor`, `navigate`).
> Eval a stage's brain: set its scenario/map env and run `doom-cli eval --algo ppo`.

---

# 🧠 Semantic channel — feeding the detections INTO the network (2026-06-06)

The agent's policy only ever saw raw pixels (+ depth/spatial/health) and had to INFER what a
region was; it did not know "enemy" vs "item" vs "door" (those were human-overlay only). We added
a **semantic obs channel** (`SEMANTIC_CHANNEL=1`, brain tag `_se`): an 84×84 map where each
on-screen object is painted by category code (enemy/weapon/health/…) from the labels detector,
plus every door projected from the WAD. The network now perceives **what is where**.

## Controlled A/B (the rigorous test)

Two **fresh 1M** PPO runs on MAP01, **identical config + same seed (42)**, only `SEMANTIC_CHANNEL`
differs. Eval: 10 episodes, T=0.5.

| Metric | Baseline (no semantic) | **Semantic** | Δ |
|---|---|---|---|
| map explored | 16% (156 cells) | **25% (235)** | **+56%** |
| exit progress (how close to exit) | 17% | **35%** | **2×** |
| kills / episode | 6.1 | **10.3** | +69% |
| shooting accuracy | 7% | **13%** | ~2× |
| shots landed / ep | 19.2 | **34.5** | +80% |
| enemies seen / ep | 9.3 | **11.9** | +28% |

**Conclusion: feeding categories into the network causes a real, substantial gain** — the first
clear feature win on full MAP01. Both still 0% exit (neither reaches the end; the semantic brain
gets ~2× closer). Caveat: single seed + 10 eval eps (multi-seed validation in progress).

## Richer, non-binary metrics (replacing exit-rate as the headline)

exit-rate is binary and harsh. `rl.eval` now prints a **"what happened (per episode)"** block:
- **exit progress** — how close to the exit, now computed even WITHOUT reaching it by reading the
  EXIT linedef straight from the WAD (`doom.wad_doors.map_exit`; metric-only, no reward change).
- **enemies seen** (distinct), **shots fired** (+ landed/accuracy), **hits taken** (+HP damage),
  **heals consumed** (+HP) — the real story of a run.

## What did NOT help: demo retrieval (nearest-neighbour imitation)

`--recall` replays the human's action from the most-similar demo frame (raw-pixel descriptor, and
a learned-autoencoder embedding). Built + tested, but it does **not** improve exit on MAP01: the
agent drifts off the demos' path, so the nearest demo frame isn't the right action for its actual
situation (classic BC distribution-shift / DAgger problem). Reusable with many more demos.

> Reproduce: fresh A/B → `SEMANTIC_CHANNEL=0/1 CAMPAIGN=1 MAPS=MAP01 python -m rl.train --fresh
> --timesteps 1000000`, then `... python -m rl.eval --episodes 10 --temperature 0.5`.
