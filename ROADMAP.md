# 🗺️ HeLLMind — Functional TODO

Honest status of every item: ✅ done · 🟡 partial (exists, needs work) · ❌ not built.
The point of this file: the agent's **quality** (it ignores enemies, gets stuck on doors) is
the real gap — most "cognition" machinery already exists; it needs a stronger agent under it.

---

## 🔴 P0 — Make the agent ACTUALLY PLAY (the real problem)

Watching the bot: it ignores enemies, doesn't shoot, bangs on closed doors. Root causes found:

- [x] **`watch` shows the FROZEN argmax** ✅ DONE — `watch` now defaults to tempered T=0.5 so
  you see the real learned policy (`--temperature 0` for raw argmax).
- [x] **Doors** ✅ DONE — `AUTO_USE` holds USE every frame so doors open / switches fire on
  contact (no longer a dead end). On by default.
- [x] **Decouple combat & exploration** 🟡 DONE (reward-level) — gated by enemy visibility:
  combat focus when enemies on screen, exploration focus when clear. Per-mode telemetry
  (`combat_engagement`) lets the coach tune each regime separately. ❌ Still ONE network —
  the true two-policy split (champion architecture) is a later upgrade.
- [ ] **Undertraining.** 50k steps is nothing. This is the compute gap — needs the long `auto`
  / Colab runs, not a code change.

---

## 🟠 P1 — Prove it works (benchmark + multi-seed) — *from your roadmap*

- [x] **`doom-cli benchmark`** ✅ DONE — no-arg ablation (baseline→rnd→memory→full × seeds),
  writes `results/{benchmark.json,.csv,.md}` with mean±std. *(plots/ still TODO)* Matrix:
  | Config | PPO | RND | Memory | Coach | Knowledge |
  |---|---|---|---|---|---|
  | Baseline | ✅ | | | | |
  | +RND | ✅ | ✅ | | | |
  | +Memory | ✅ | ✅ | ✅ | | |
  | Full | ✅ | ✅ | ✅ | ✅ | ✅ |
  Metrics: exploration %, kills, survival, exit-rate, reward, unique coverage.
- [x] **Multi-seed eval** ✅ DONE — `benchmark` runs configurable seeds with mean ± std.
- [ ] **Research dashboard** ❌ — `doom-cli report` → HTML with the curves.

---

## 🟡 P2 — Exploration

- [x] **Behavioral cloning** ✅ — `record_demo` → `bc` → `auto` (Phase 2 script ready).
- [x] **Frontier intelligence** ✅ DONE — goal sampling now weights distance/visits × edge-bonus
  (boundary of explored region) × aging-decay; stale frontiers are pruned.
- [x] **Automatic goal discovery** ✅ DONE — `DISCOVERY_REWARD` pays the first sighting of each
  new object (keys/weapons/powerups/new monsters) per episode via the labels buffer.

---

## 🟡 P3 — Memory

- [x] **Episodic memory** ✅ — events store situation→action→result (weapon/region/nearest_enemy).
- [x] **Semantic recall** ✅ — `doom-cli recall "revenant"` (by keyword/enemy/region) works.
- [x] **Long-term knowledge tiers** ✅ DONE — `writer/knowledge.py` + `doom-cli knowledge`
  present Facts (bestiary) / Hypotheses (open) / Validated (confirmed + learned_config).

---

## 🟡 P4 — Coach (the self-improvement loop)

- [x] **Structured hypotheses** ✅ — `hypothesize` → `experiment` → verdict → `learned_config`.
- [x] **Auto-chain it inside `auto`** ✅ DONE — every auto iteration now records its reward
  change + keep/revert verdict into the experiment registry (single-seed, so NOT auto-adopted;
  multi-seed `experiment` stays the validation path).
- [x] **Rollback system** ✅ DONE — structured `before/change/after/result/kept` records in
  `rollback.jsonl` (`writer/rollback.py`, `doom-cli rollback`) — every adjustment reversible.
- [x] **Experiment registry** ✅ — SQLite `experiments` table + `db query --experiments`,
  populated by both `experiment` (validated) and `auto` (kept/reverted trail).

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

## ✅ Wave 1 — DONE (this pass)

- `watch --temperature` (see the real agent) · combat/exploration decoupling (reward-level) ·
  per-mode telemetry so the coach tunes combat & exploration separately · `doom-cli benchmark`
  (P1 multi-seed ablation) · auto tunes 5 more knobs incl. ENT_COEF (argmax-collapse) ·
  Obsidian Autonomy Log shows the combat regime · fixed 4 blocking bugs (EPISODE_TIMEOUT
  float, gif, db help, doom-cli maps) + tech debt.

## ✅ Wave 2 — DONE (this pass)

- AUTO_USE (doors open on contact — the gameplay fix) · frontier intelligence (aging + edge
  prioritization + pruning) · automatic goal discovery (reward first sighting of keys/weapons/
  powerups/new monsters via the labels buffer).

## ✅ Wave 3 — DONE (this pass)

- Long-term knowledge tiers (`doom-cli knowledge`: facts/hypotheses/validated) · auto-chain
  (every auto iteration logged to the experiment registry with its keep/revert verdict).

## ▶️ Next waves
- **Wave 4 (P6+P7):** ablation report plots · `doom-cli report` (HTML) · GIFs (initial vs
  trained) · `docs/research/` paper · assisted mode (your live-feedback loop).
- **Always-on:** long `auto`/Colab runs for the compute gap (the real exit-rate lever).
