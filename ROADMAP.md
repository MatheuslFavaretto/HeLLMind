# 🗺️ HeLLMind — Functional TODO

Honest status of every item: ✅ done · 🟡 partial (exists, needs work) · ❌ not built.
The point of this file: the agent's **quality** (it ignores enemies, gets stuck on doors) is
the real gap — most "cognition" machinery already exists; it needs a stronger agent under it.

---

## 🔴 P0 — Make the agent ACTUALLY PLAY (the real problem)

Watching the bot: it ignores enemies, doesn't shoot, bangs on closed doors. Root causes found:

- [ ] **`watch` shows the FROZEN argmax, not the learned policy.** 🟡 `eval` has `--temperature`
  but `watch` is pure argmax → you're literally watching the collapsed (passive) policy.
  **Fix:** add `--temperature` to `watch` so you see the real agent. *(quick win, do first)*
- [ ] **Doors have no reward signal.** USE exists only as the `FWD+USE` action (1 of 11), and
  nothing rewards opening a door → no gradient to learn it. **Fix options:** (a) auto-press USE
  every frame, (b) reward area-progress through a door, (c) add USE to more action combos.
- [ ] **Decouple combat & exploration (champion architecture).** 🟡 ❌ One network juggles both.
  Arnold (ViZDoom winner) used **separate nav + combat policies** with a gating signal
  (enemy-on-screen from the labels buffer, which we already have). Build a 2-head / 2-policy
  split routed by `is_enemy_visible`.
- [ ] **Undertraining.** 50k steps is nothing. This is the compute gap — needs the long `auto`
  / Colab runs, not a code change.

---

## 🟠 P1 — Prove it works (benchmark + multi-seed) — *from your roadmap*

- [ ] **`doom-cli benchmark` (a.k.a. `performance`)** ❌ — no-arg command that runs the ablation
  matrix simply and writes `results/{benchmark.csv,.json,.md,plots/}`:
  | Config | PPO | RND | Memory | Coach | Knowledge |
  |---|---|---|---|---|---|
  | Baseline | ✅ | | | | |
  | +RND | ✅ | ✅ | | | |
  | +Memory | ✅ | ✅ | ✅ | | |
  | Full | ✅ | ✅ | ✅ | ✅ | ✅ |
  Metrics: exploration %, kills, survival, exit-rate, reward, unique coverage.
- [ ] **Multi-seed eval** 🟡 — `experiment` already runs seeds 42,123. Generalize to 5 seeds with
  mean ± std ± CI so wins aren't luck.
- [ ] **Research dashboard** ❌ — `doom-cli report` → HTML with the curves.

---

## 🟡 P2 — Exploration

- [x] **Behavioral cloning** ✅ — `record_demo` → `bc` → `auto` (Phase 2 script ready).
- [ ] **Frontier intelligence** 🟡 — Go-Explore exists; add frontier **scoring + aging +
  prioritization** so it stops returning to useless areas.
- [ ] **Automatic goal discovery** ❌ — detect new doors/corridors/rooms/keys/switches and reward
  discovery (progress-guided exploration). Use the ViZDoom labels/objects buffers we already read.

---

## 🟡 P3 — Memory

- [x] **Episodic memory** ✅ — events store situation→action→result (weapon/region/nearest_enemy).
- [x] **Semantic recall** ✅ — `doom-cli recall "revenant"` (by keyword/enemy/region) works.
- [ ] **Long-term knowledge tiers** 🟡 — split Facts / Hypotheses / Validated. We have hypotheses
  + lessons + learned_config; formalize the 3 tiers explicitly.

---

## 🟡 P4 — Coach (the self-improvement loop)

- [x] **Structured hypotheses** ✅ — `hypothesize` → `experiment` → verdict → `learned_config`.
- [ ] **Auto-chain it inside `auto`** ❌ — today the falsifiable cycle is MANUAL (or via
  `research`). `auto` only tunes reward + reverts. Wire hypothesize→experiment→adopt into `auto`.
- [x] **Rollback / keep-if-improved** ✅ — auto reverts regressions (see `timeline`).
- [x] **Experiment registry** ✅ — SQLite `experiments` table + `db query --experiments`.

---

## ⚪ P5 — Platform (later)

- [ ] Multiple ViZDoom scenarios (MyWayHome, Deadly Corridor, Health Gathering) — use the
  **[official ViZDoom tutorial](https://vizdoom.cs.put.edu.pl/tutorial)** as the reference.
- [ ] Modular agent (Environment / Brain / Memory / Coach / Knowledge as swappable modules).
- [ ] Plugin system (`MemoryPlugin`, `CoachPlugin`).
- [ ] **LLM backend abstraction** 🟡 — decouple from Ollama (so llama.cpp/others are drop-in).

---

## ⚪ P6 — Scientific evidence (later)

- [ ] Auto learning curves (reward/exploration/kills/exit per checkpoint) — 🟡 `progress` exists.
- [ ] Ablation studies (no-memory / no-RND / no-coach / no-automap / no-depth).
- [ ] Internal paper `docs/research/` (method, results, limitations, future work).

---

## 🟢 P7 — Make GitHub impressive (later)

- [ ] GIFs: initial vs trained, exploration, combat, evolution — 🟡 `gif` works now (bug fixed).
- [ ] `doom-cli report` HTML dashboard.
- [x] Interactive-ish timeline ✅ — `doom-cli timeline` (could add hypotheses/changes columns).

---

## 🆕 New: Assisted mode (your request)

- [ ] **`doom-cli assist`** ❌ — opens the game window, you watch live, and there's a tight
  feedback loop: you report what's wrong → it logs your note as a behavior flag/hypothesis →
  you/the coach adjust → repeat. (A human-in-the-loop version of the knowledge loop.)

---

## ▶️ Recommended order

1. `watch --temperature` (5 min) — so you finally SEE the real agent, not the frozen argmax.
2. `doom-cli benchmark` — the proof-it-works matrix (P1) — your highest-value roadmap item.
3. Door reward + combat/exploration split (P0) — the actual gameplay fix.
4. Assisted mode (your feedback loop) + long `auto` runs for the compute gap.
