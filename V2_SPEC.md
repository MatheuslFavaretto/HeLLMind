# 🔭 HeLLMind V2 — Spec & Plan

> Grounded in a real audit of this repo's DB, Obsidian vault, and code (not a generic spec).
> Written to be honest: where V1 works, where it's broken, and what V2 should actually be.

---

## 0. The one-paragraph thesis

V1 proved the *cognition scaffolding* (memory, hypotheses, rollback, knowledge tiers, a
self-tuning loop). But it wraps **10.8k lines / 57 modules** around an agent whose best-ever
result is **11% map explored, 0% exit-rate, 80% death-rate**. The machinery outgrew the
agent. **V2 is not more features — it's a stronger RL core + real compute + a leaner, better-
integrated system.** We keep the best of the cognition layer (Voyager-style) and cut the rest.

---

## 1. Honest current-state audit (from the DB/vault)

| Signal | Reality (measured) |
|---|---|
| Best run ever | iter-5: explored **11%**, exit **0%**, kills 1.7, **death 80%** |
| Validated experiments | 5 · Hypotheses 6 · Lessons 15 |
| Code size | 10,849 lines, 57 modules (21 rl / 29 writer / 7 doom) |
| Compute use | **CPU only** (M5 GPU/MPS unused); n_envs **4 of 10 cores** |
| Integration | many modules invoked ad-hoc by subprocess; no single source of truth |

**Diagnosis:** the bottleneck was never features — it's (a) **sample efficiency** (PPO is
sample-hungry, we never had the FLOPS) and (b) **death-rate 80%** (the agent dies before it
can explore/finish). Everything else is secondary.

---

## 2. Correcting the brief (important — some premises are wrong)

You wrote a few things that aren't accurate about the current code. Honest corrections:

| Your premise | Reality |
|---|---|
| "sem PPO/GRPO implementado" | **PPO IS implemented** (Stable-Baselines3, `rl/train.py`). It trains, checkpoints, resumes. |
| "sem training loop contínuo" | There IS a loop (`doom-cli auto`): train → eval → tune → repeat. It's *chunked*, not *streaming*. |
| "sem replay buffer" | **Correct — but by design.** PPO is *on-policy*: it has no replay buffer. A replay buffer means switching to *off-policy* (DQN/Rainbow) or *model-based* (DreamerV3). That's a real V2 decision (§4). |
| "sem policy update automático" | PPO updates the policy every rollout automatically. What's missing is *continuous/async* updates. |
| "troque SQLite pelo Vector DB" | They do **different jobs**. SQLite = structured queryable logs; Vector DB = semantic similarity. Your own spec (§3.9) lists **both**. V2 **adds** a vector DB; it doesn't replace SQLite. |

**Why this matters:** the highest-leverage V2 change isn't "add a replay buffer" as a bolt-on
— it's **choosing an RL algorithm that's sample-efficient**, which then *brings* a replay
buffer with it.

---

## 3. 🗑️ What to REMOVE (you asked) — cut to integrate

The cognition layer has redundant/overlapping writers. Proposed cuts/merges:

- **Merge** `writer/process_run.py`, `writer/reflect.py`, `writer/note_writer.py`,
  `writer/compare_runs.py`, `writer/suggest.py` → one `writer/documenter.py` (one LLM path).
- **Remove `eureka`** (LLM evolves reward functions) — unproven, heavy, and it fights the
  coach's own reward tuning. One reward-tuner, not two.
- **Remove `research_agent`** as a separate flow — fold its behavior→hypothesis→experiment
  chain into `auto` (already half-done in V1).
- **Demote** `make_demo.py` / `bestiary_chart.py` to optional scripts (not core).
- **Collapse** the reward-shaping zoo: V1 has ~12 shaping terms (coverage, frontier, RND,
  engagement, discovery, exit-prox, go-explore, bestiary, weapon-variety…). Keep **4**:
  curiosity (RND), frontier/coverage (one), exit-proximity, death-penalty. The rest is
  reward-hacking surface area. *(This is the single biggest simplification.)*

Net: target **~6k lines**, one documenter, one reward-tuner, 4 shaping terms.

---

## 4. The core decision: which RL for V2?

The compute gap + 80% death-rate both point to **sample efficiency**. Three options:

| Option | Replay buffer | Sample-eff | Risk | Verdict |
|---|---|---|---|---|
| **Keep PPO** (cleanrl-style rewrite) | no | low | low | baseline only |
| **Rainbow DQN** (off-policy, discrete) | ✅ | medium | medium | **safe upgrade** — proven on Doom |
| **DreamerV3** (model-based, world model) | ✅ | **highest** | high | **the bet** — learns from imagination |

**Recommendation:** ship **Rainbow DQN** as the new default (gives you the replay buffer +
continuous off-policy updates you want, and it's far more sample-efficient than PPO on
discrete Doom), and **prototype DreamerV3** in parallel as the high-ceiling research track
(it directly attacks the compute wall by learning a world model and training in imagination).
Keep PPO as a benchmark baseline. Reference impls: cleanRL (single-file clarity), SB3 +
sb3-contrib (Rainbow pieces), `danijar/dreamerv3`.

---

## 5. What worked here + what to borrow from the references

| Project | Borrow |
|---|---|
| **this repo (V1)** | the cognition scaffolding: memory, rollback, knowledge tiers, honest tempered eval, multi-seed benchmark — **keep all of it** |
| **cleanrl** | single-file, readable algo implementations — rewrite the RL core this way (no more 10-module sprawl) |
| **stable-baselines3 / sb3-contrib** | robust, tested algos + VecEnv + Rainbow components |
| **ViZDoom** | the buffers we under-use: labels (bbox), depth, automap (minimap), objects (HUD) → §8 overlays |
| **Voyager** ⭐ | the north star for the cognition layer: an LLM that builds a **skill library** + curriculum + self-verification. Our coach should grow *reusable skills*, not just tweak reward knobs |
| **dreamerv3** ⭐ | world model + replay + train-in-imagination = the sample-efficiency answer |
| **ray / rllib** | distributed rollout workers + central trainer (the §7 scaling, when we outgrow one machine) |
| **langgraph** | model the coach as an explicit **graph** (observe→hypothesize→experiment→validate→adopt) instead of ad-hoc Python |
| **AlphaZero (1712.01815)** | self-play + search — a *future* extension (multi-agent Doom) |
| **procgen** | the generalization test — train on some maps, eval on held-out ones |

---

## 6. V2 architecture

```
┌──────────────── ENVIRONMENT (ViZDoom, N async workers) ─────────────────┐
│  frame · game-vars (HUD) · labels (bbox) · automap (minimap) · done      │
└───────────────────────────────────┬─────────────────────────────────────┘
                                     ▼  transitions
┌──────────────── REPLAY BUFFER (local shards; Redis later) ───────────────┐
│           (state, action, reward, next_state, done)                      │
└───────────────────────────────────┬─────────────────────────────────────┘
                                     ▼  sampled batches
┌──── LEARNER (GPU/MPS) ────┐   ┌──── COACH (LangGraph) ──────────────────┐
│ Rainbow DQN  (+ Dreamer   │◀──│ behaviour → hypothesis → experiment →   │
│ world-model track)        │   │ validate → adopt · skill library (Voyager)│
└──────────┬────────────────┘   └──────────────────┬──────────────────────┘
           ▼ policy checkpoints                     ▼ reads/writes
┌──── INFERENCE (watch / auto) ──┐   ┌──── MEMORY ─────────────────────────┐
│ overlays: HUD · bbox · minimap │   │ SQLite (structured) + Vector DB     │
└────────────────────────────────┘   │ (semantic) + graph (causal, later)  │
                                      └─────────────────────────────────────┘
```

---

## 7. Compute — use ALL CPU + GPU (you asked)

1. **GPU/MPS now:** the M5 GPU is idle. Set PyTorch device to **`mps`** (Apple Silicon) for
   the learner. Worth far more for Rainbow/Dreamer (bigger nets) than for V1's tiny PPO CNN.
2. **All CPU cores:** default `N_ENVS` to `cpu_count − 2` (8 on your M5), not 4. The `--fast`
   flag already does this — make it the default for `auto`.
3. **Separate inference from training** (different processes) so rollout workers saturate the
   CPU while the learner saturates the GPU — the classic async-RL split (ray/rllib pattern).
4. **Cloud burst (later):** ray for N VizDoom workers + one central GPU trainer.

---

## 8. Structured overlays (you asked) — for `watch`

ViZDoom already gives us everything; we just render it:
- **HUD**: health/ammo from game-vars (we feed these to the policy already → just draw them).
- **Enemy bounding boxes**: the labels buffer has per-actor screen bboxes (we use it for
  `combat_engagement` → draw the boxes).
- **Minimap**: the automap buffer (already an obs channel → show it in a corner).

Deliverable: `doom-cli watch --overlay` renders the frame + HUD + bboxes + minimap (great for
demos *and* debugging "why didn't it shoot that enemy").

---

## 9. Memory: keep SQLite, ADD a vector DB

- **SQLite** stays for structured logs/runs/experiments (it's fast, zero-dep, works).
- **Vector DB** for *semantic* memory (`recall "revenant"` → similar past situations by
  embedding). Options that fit a local-first project: **sqlite-vec** (no new service) or
  **chromadb** (richer). Embeddings via the local model.
- **Graph DB** (causal memory) — defer to V3.

---

## 10. Phased plan

- **Phase 0 — Integrate & cut (1 wk):** the §3 removals; one documenter; one config source of
  truth; enable MPS + default N_ENVS to cores. *No new ML — just make V1 coherent.*
- **Phase 1 — RL core (2 wk):** cleanrl-style Rainbow DQN with a real replay buffer +
  continuous updates; PPO kept as the benchmark baseline; run `doom-cli benchmark`.
- **Phase 2 — Death-rate first (1 wk):** the measured killer is dying (80%). Reward + curriculum
  focused on *survival* before *exit*. Target: death-rate < 40%.
- **Phase 3 — Overlays + vector DB (1 wk):** §8 + §9.
- **Phase 4 — Coach as a graph (2 wk):** LangGraph coach + a Voyager-style skill library.
- **Phase 5 — DreamerV3 track (research):** world model, train-in-imagination.
- **Phase 6 — Distributed (when needed):** ray workers + central GPU trainer.

**Definition of done for "V2 works":** on `doom-cli benchmark`, the full agent **beats the
PPO baseline** AND exit-rate > 0 on at least one map. Until then, no new features.

---

## 11. What we KEEP from V1 (don't throw the baby out)

Honest, tempered eval · multi-seed benchmark (+score+HTML) · rollback log · knowledge tiers ·
SQLite memory · the Claude-style shell · the closed-loop coach concept. These are genuinely
good and rare. V2 makes the **agent under them** finally worth the scaffolding.
