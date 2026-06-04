# 🗺️ HeLLMind — Functional TODO

Honest status of every item: ✅ done · 🟡 partial (exists, needs work) · ❌ not built.
The point of this file: the agent's **quality** (it ignores enemies, gets stuck on doors) is
the real gap — most "cognition" machinery already exists; it needs a stronger agent under it.

---

## 🔴 P0 — Make the agent ACTUALLY PLAY (the real problem)

Watching the bot: it ignores enemies, doesn't shoot, bangs on closed doors. Root causes found:

- [x] **`watch` shows the FROZEN argmax** ✅ DONE — `watch` now defaults to tempered T=0.5 so
  you see the real learned policy (`--temperature 0` for raw argmax).
- [ ] **Doors have no reward signal.** USE exists only as the `FWD+USE` action (1 of 11), and
  nothing rewards opening a door → no gradient to learn it. **Fix options:** (a) auto-press USE
  every frame, (b) reward area-progress through a door, (c) add USE to more action combos.
  *(next wave — the remaining P0 gameplay fix)*
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

## ✅ Wave 1 — DONE (this pass)

- `watch --temperature` (see the real agent) · combat/exploration decoupling (reward-level) ·
  per-mode telemetry so the coach tunes combat & exploration separately · `doom-cli benchmark`
  (P1 multi-seed ablation) · auto tunes 5 more knobs incl. ENT_COEF (argmax-collapse) ·
  Obsidian Autonomy Log shows the combat regime · fixed 4 blocking bugs (EPISODE_TIMEOUT
  float, gif, db help, doom-cli maps) + tech debt.

## ▶️ Next waves

- **Wave 2 (P0+P2):** door reward (the gameplay fix) · frontier intelligence
  (scoring/aging/prioritization) · automatic goal discovery (doors/keys/switches).
- **Wave 3 (P4+P3):** auto-chain hypothesize→experiment inside `auto` · long-term knowledge
  tiers (facts/hypotheses/validated).
- **Wave 4 (P6+P7):** ablation report polish + plots · `doom-cli report` (HTML) ·
  GIFs (initial vs trained) · `docs/research/` paper · assisted mode.
- **Always-on:** long `auto`/Colab runs for the compute gap (the real exit-rate lever).
