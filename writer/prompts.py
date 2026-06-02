"""System prompts and message builders for the LLM that documents training.

Quality decision: we do NOT dump raw JSON at the model (a small model gets lost and
hallucinates). Instead we pre-digest the metrics into a readable "fact sheet", with
the deltas vs. the previous checkpoint already computed and trend arrows. This
directly fights the "notes don't reflect reality" problem.
"""
from typing import Dict, List, Optional

# CHECKPOINT note system prompt. Fixed -> good candidate for prompt caching.
CHECKPOINT_SYSTEM = """\
You are a Reinforcement Learning researcher documenting, in English, the training of
a PPO agent (CnnPolicy) playing Doom (ViZDoom).

You receive a pre-computed METRICS REPORT (current window values and the change vs.
the previous checkpoint). Write a checkpoint note for an Obsidian vault following
STRICT RULES:

GOLDEN RULE — FIDELITY: use ONLY the numbers in the report. Do NOT invent facts, do
not assume events that aren't in the data, do not exaggerate. If a metric didn't
change meaningfully, say it stayed stable. If the data is thin/few episodes, say so
explicitly instead of fabricating a narrative.

1. behavior_change: interpret WHAT CHANGED IN THE BEHAVIOR, anchoring EVERY claim to a
   number from the report (e.g., "shooting accuracy rose from 18% to 31%, indicating
   better aim"). No claim without a number to back it. On the FIRST checkpoint there is
   no comparison — describe the starting point, don't invent progress.
2. evidence: 3-5 bullets, each a sentence of YOUR OWN citing a number (accuracy,
   kills/ep, damage, distance, entropy, success). Do NOT copy the report lines
   verbatim, with no "- " prefix and no annotations like "(1st checkpoint)".
3. RL concepts: PREFER linking concepts that already exist (list provided). Only create
   a new concept if it is genuine and reusable (e.g., "Reward Shaping", "Policy
   Entropy", "Exploration vs Exploitation", "Sample Efficiency"). A concept name is
   ONLY the canonical term, in Title Case — NEVER include a URL, link, markdown,
   numbers, values, or trend words ("up", "down", "from 0.9 to 0.7") in the name.
   Right: "Policy Entropy". Wrong: "Policy Entropy down from 0.9 to 0.7".
4. Be concise and honest. No filler.

Respond ONLY in the requested structured format. Concept names short and stable
(Title Case) — they become note titles and wikilinks.\
"""

# CONCEPT note system prompt (generated on demand when a concept is new).
CONCEPT_SYSTEM = """\
You are an RL researcher writing, in English, a reusable CONCEPT note for an Obsidian
vault. The note must be timeless (not tied to a specific checkpoint), explain the
concept clearly and objectively, and indicate how it shows up while training an agent
to play Doom. Respond only in the requested structured format.\
"""


def _trend(cur: float, prev: Optional[float], pct: bool = False, unit: str = "") -> str:
    """'31% (up from 18%)' / '12.0 (down from 15.0)' / '5.0 (stable)' — readable delta."""
    def fmt(v: float) -> str:
        return f"{v:.0%}" if pct else (f"{v:,.1f}{unit}" if v % 1 else f"{v:,.0f}{unit}")

    if prev is None:
        return f"{fmt(cur)} (1st checkpoint)"
    diff = cur - prev
    denom = abs(prev) if abs(prev) > 1e-6 else 1.0
    if abs(diff) / denom < 0.05:
        return f"{fmt(cur)} (stable)"
    arrow = "up" if diff > 0 else "down"
    return f"{fmt(cur)} ({arrow} from {fmt(prev)})"


def build_checkpoint_user_message(
    snapshot: Dict,
    previous: Optional[Dict],
    existing_concepts: List[str],
    button_names: List[str],
) -> str:
    """Build a readable FACT SHEET (not raw JSON) so the model doesn't invent."""
    s, p = snapshot, (previous or {})

    def g(d: Dict, k: str) -> float:
        return float(d.get(k, 0.0))

    def pv(k: str) -> Optional[float]:
        return float(p[k]) if (previous and k in p) else None

    cov = s.get("map_coverage", {}) or {}
    weapons = s.get("weapons_used", {}) or {}
    weapons_txt = ", ".join(f"{k}={v:.0%}" for k, v in weapons.items()) or "n/a"
    dist = s.get("action_distribution", {}) or {}
    top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_txt = ", ".join(f"{k} {v:.0%}" for k, v in top) or "n/a"

    lines = [
        f"# Checkpoint report @ {int(g(s,'num_timesteps')):,} timesteps",
        f"Window: {int(g(s,'episodes'))} episodes, {int(g(s,'steps_in_window'))} steps.",
    ]
    if s.get("map"):
        lines.append(f"Map: {s['map']} | success rate (completion): "
                     f"{_trend(g(s,'success_rate'), pv('success_rate'), pct=True)}")
    lines += [
        "",
        "## Performance",
        f"- Mean reward/episode: {_trend(g(s,'mean_reward'), pv('mean_reward'))}",
        f"- Kills/episode: {_trend(g(s,'kills_per_episode'), pv('kills_per_episode'))}",
        f"- Mean episode length: {_trend(g(s,'mean_episode_length'), pv('mean_episode_length'))} steps",
        "",
        "## Aim (hits vs misses)",
        f"- Shots fired: {int(g(s,'shots_fired'))} | hits: {int(g(s,'shots_hit'))} | misses: {int(g(s,'shots_missed'))}",
        f"- Accuracy: {_trend(g(s,'shooting_accuracy'), pv('shooting_accuracy'), pct=True)}",
        f"- Damage dealt: {_trend(g(s,'damage_dealt'), pv('damage_dealt'))} | damage taken: {_trend(g(s,'damage_taken'), pv('damage_taken'))}",
        "",
        "## Exploration / path",
        f"- Distance/episode: {_trend(g(s,'distance_per_episode'), pv('distance_per_episode'))} units",
        f"- Cells visited: {int(g(s,'cells_visited'))} (~{cov.get('explored_fraction',0.0):.0%} of the traversed area)",
        f"- Weapons used (fraction of time): {weapons_txt}",
        "",
        "## Policy",
        f"- Action entropy (norm.): {_trend(g(s,'action_entropy_normalized'), pv('action_entropy_normalized'))}",
        f"- Most used actions: {top_txt}",
        f"- Mean health: {g(s,'mean_health'):.0f} | mean ammo: {g(s,'mean_ammo'):.0f}",
        "",
        "## Existing concepts in the vault (PREFER linking these)",
        ("- " + "\n- ".join(existing_concepts)) if existing_concepts else "(none yet)",
    ]
    return "\n".join(lines)


def build_concept_user_message(name: str, hint: str) -> str:
    return (
        f"Write the concept note for: '{name}'.\n"
        f"Context/observation that motivated it: {hint}"
    )


# --------------------------- Run synthesis (A) ---------------------------
RUNSTORY_SYSTEM = """\
You are an RL researcher writing, in English, the SYNTHESIS of a whole PPO training run
on Doom. You receive a timeline (one summary per checkpoint).

Fill TWO distinct fields, without repeating one in the other:
- **narrative**: 2 to 4 PARAGRAPHS of flowing prose telling the ARC of the learning end
  to end (how behavior evolved: when it learned to aim, to move, to survive; plateaus
  and regressions; what the numbers suggest about the policy). NO bullets here, do NOT
  list checkpoint by checkpoint — it's an interpretive narrative.
- **milestones**: SHORT bullets, one per relevant milestone, each citing the approximate
  step (e.g., "~25k: accuracy jumps from 8% to 15%"). Only the turning points.

RULE: anchor everything to the timeline numbers; do not invent events. If the signal is
noisy/short, say so. Respond ONLY in the requested structured format.\
"""


def build_run_story_user_message(
    run_name: str, snapshots: List[Dict], concepts: List[str]
) -> str:
    lines = [
        f"Run: {run_name} — {len(snapshots)} checkpoints.",
        "",
        "Timeline (each line is a checkpoint, in order):",
    ]
    for s in snapshots:
        lines.append(
            f"- {int(s.get('num_timesteps', 0)):,} steps: "
            f"reward {float(s.get('mean_reward', 0)):.1f}, "
            f"accuracy {float(s.get('shooting_accuracy', 0)):.0%}, "
            f"kills/ep {float(s.get('kills_per_episode', 0)):.1f}, "
            f"success {float(s.get('success_rate', 0)):.0%}, "
            f"entropy {float(s.get('action_entropy_normalized', 0)):.2f}, "
            f"dist/ep {float(s.get('distance_per_episode', 0)):.0f}"
        )
    lines += ["", "Concepts already documented in this run:",
              ("- " + "\n- ".join(concepts)) if concepts else "(none)"]
    return "\n".join(lines)


# ------------------------- Run comparison (B) -------------------------
COMPARE_SYSTEM = """\
You are an RL researcher comparing, in English, TWO OR MORE training runs of the same
agent on Doom (e.g., with vs. without a reward shaping change). You receive a per-run
metrics summary (final, best, mean). State objectively which run was better and WHY,
citing the numbers. If the difference is small/within noise, say 'tie'. Do not invent.
Respond ONLY in the requested structured format.\
"""


# --------------------------- Reward suggestions (Phase 6) ---------------------------
SUGGEST_SYSTEM = """\
You are an RL researcher proposing, in English, small changes to the reward-shaping
weights of a PPO agent on Doom, based on observed behavior. You may ONLY tune these
knobs: hit_reward, miss_penalty, damage_taken_penalty, death_penalty. Propose 1-3
tweaks, each with the new value and a reason grounded in the numbers.

Guidance: low accuracy -> consider raising hit_reward or miss_penalty (but keep
miss_penalty < hit_reward to avoid a passive agent); many low-HP deaths -> consider
raising damage_taken_penalty or death_penalty. Keep changes modest (e.g. ≤ 2x).
These are SUGGESTIONS for a human to approve — never assume they are applied. Respond
ONLY in the requested structured format.\
"""


def build_suggest_user_message(stats: Dict, weights: Dict) -> str:
    return "\n".join([
        "Current reward weights:",
        f"- hit_reward = {weights.get('hit_reward')}",
        f"- miss_penalty = {weights.get('miss_penalty')}",
        f"- damage_taken_penalty = {weights.get('damage_taken_penalty')}",
        f"- death_penalty = {weights.get('death_penalty')}",
        "",
        "Observed behavior (across runs):",
        f"- Shooting accuracy: {stats.get('shooting_accuracy', 0.0):.0%}",
        f"- Death rate: {stats.get('death_rate', 0.0):.0%} "
        f"({int(stats.get('deaths', 0))} deaths)",
        f"- Low-HP deaths (<30): {stats.get('low_hp_death_rate', 0.0):.0%}",
        f"- Mean health just before death: {stats.get('mean_health_at_death', 0.0):.1f}",
    ])


# --------------------------- Lessons / reflection (Phase 4) ---------------------------
LESSONS_SYSTEM = """\
You are an RL researcher extracting reusable LESSONS, in English, from aggregated
outcomes of many Doom episodes ACROSS runs (deaths, successes, timeouts and their
context). You receive a pre-computed statistics report. Write 3-6 concrete, actionable
lessons — especially failure patterns (e.g. "the agent dies in corridors below 25 HP").

RULE: ground EVERY lesson in the numbers from the report; cite them in `evidence`. Do
not invent. If the data is thin, say so and give fewer, hedged lessons. Respond ONLY
in the requested structured format.\
"""


def build_lessons_user_message(stats: Dict) -> str:
    lines = [
        f"Aggregated over {int(stats.get('total', 0))} episode events "
        f"({int(stats.get('runs', 0))} run(s)):",
        "",
        f"- Deaths: {int(stats.get('deaths', 0))} "
        f"({stats.get('death_rate', 0.0):.0%}) | "
        f"Successes: {int(stats.get('successes', 0))} | "
        f"Timeouts: {int(stats.get('timeouts', 0))}",
        f"- Mean health just before death: {stats.get('mean_health_at_death', 0.0):.1f}",
        f"- Mean ammo just before death: {stats.get('mean_ammo_at_death', 0.0):.1f}",
        f"- Low-HP deaths (health < 30): {stats.get('low_hp_death_rate', 0.0):.0%} of deaths",
        f"- Mean episode length — deaths {stats.get('mean_len_death', 0.0):.0f} "
        f"vs successes {stats.get('mean_len_success', 0.0):.0f} steps",
    ]
    by_map = stats.get("deaths_by_map", {}) or {}
    if by_map:
        top = ", ".join(f"{m}={c}" for m, c in by_map.items())
        lines.append(f"- Deaths by map: {top}")
    return "\n".join(lines)


def build_comparison_user_message(labels: List[str], summaries: Dict) -> str:
    lines = ["Run comparison. Per-run metrics (final / best / mean):", ""]
    for label in labels:
        s = summaries.get(label, {})
        lines.append(f"## {label}  ({s.get('checkpoints', 0)} checkpoints, "
                     f"{int(s.get('timesteps', 0)):,} steps)")
        for key in ("mean_reward", "shooting_accuracy", "kills_per_episode",
                    "success_rate", "distance_per_episode"):
            m = s.get(key, {})
            lines.append(
                f"- {key}: final {m.get('final', 0):.3f} | "
                f"best {m.get('best', 0):.3f} | mean {m.get('mean', 0):.3f}"
            )
        lines.append("")
    return "\n".join(lines)
