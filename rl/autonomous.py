"""Autonomous supervisor — the agent runs itself: train -> evaluate -> adjust -> repeat.

This closes the autonomy loop. Each iteration trains a chunk (resuming the vault's
brain), evaluates it deterministically, scores it against the GOAL (explore + complete
+ fight), and then nudges the reward weights toward the weakest metric. A guardrail
reverts any tweak that makes the composite score worse, so it can only improve or hold.

Everything is logged into the vault (`.memory/autonomy.jsonl` + an Obsidian note), so
the run documents its own self-improvement — the heart of HeLLMind.

    python -m rl.autonomous --iterations 6 --steps 100000 --map MAP02
    python -m rl.autonomous --iterations 6 --steps 100000 --map MAP02 --fresh
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

# Force line-by-line flush so iter summaries appear immediately when backgrounded.
sys.stdout.reconfigure(line_buffering=True)

from config import Config

PY = sys.executable


# ---- the GOAL, as one number: explore the whole map, finish it, and survive/fight ----
def score(m: dict) -> float:
    """Composite goal score. Priority order (the weights): finishing > covering > aim >
    fighting.

    Every term is normalised to ~[0,1] FIRST, so the weights alone set priority. This
    matters: kills/ep is unbounded (0–5+), so the old `0.5 * kills` let a spawn-camping
    brain that farms ~4 kills score +2.0 — dwarfing a real explorer (explored=0.1 →
    +0.3). That scale bug literally told the agent "camp and kill > explore", the exact
    local optimum we kept hitting. Capping kills at 5 (diminishing past that) and scaling
    to [0,1] restores the intended ordering: max contributions become exit 4.0, explore
    3.0, aim 1.0, kills 0.5 — kills is now a tiebreaker, not the objective."""
    exit_r   = m.get("exit_rate", 0.0)                       # [0,1] (binary: finished?)
    # Partial credit for getting CLOSE to the exit (dense, fairer than the binary rate) — so
    # the agent is rewarded for progress toward finishing even before it completes once.
    exit_prog = m.get("exit_progress", 0.0)                 # [0,1]
    explored = m.get("explored_fraction", 0.0)              # [0,1]
    kills    = min(m.get("kills_per_episode", 0.0), 5.0) / 5.0  # [0,1] (capped)
    accuracy = m.get("shooting_accuracy", 0.0)             # [0,1]
    return (4.0 * exit_r + 1.5 * exit_prog + 3.0 * explored
            + 1.0 * accuracy + 0.5 * kills)


# Reward knobs the supervisor is allowed to move, with hard bounds (the guardrails).
BOUNDS = {
    # V2 Phase 0: COVERAGE cap lowered (was 4.0). With KILL_REWARD=10 the heuristic
    # must not bump exploration above 1.5 or it floods out the combat signal again.
    "COVERAGE_REWARD":       (0.0, 1.5),
    "EXIT_REWARD":           (0.0, 1500.0),
    "HIT_REWARD":            (1.0, 10.0),   # raised floor/ceil to match new combat scale
    "MISS_PENALTY":          (0.0, 0.3),
    "DAMAGE_TAKEN_PENALTY":  (0.0, 0.5),
    "DEATH_PENALTY":         (2.0, 25.0),   # raised floor/ceil
    "FRONTIER_REWARD":       (0.0, 0.2),
    "EPISODE_TIMEOUT":       (1050, 8400),
    "ENGAGEMENT_REWARD":     (0.0, 0.2),    # raised ceil (was 0.1)
    "ENT_COEF":              (0.005, 0.08),  # PPO un-freeze lever
    "DQN_EPS_FINAL":         (0.02, 0.3),   # QR-DQN un-freeze lever (ε-greedy floor)
    "RND_SCALE":             (0.0, 0.5),    # capped: curiosity must not dominate combat
    "GOEXPLORE_GOAL_PROB":   (0.0, 0.8),
    "COMBAT_EXPLORE_FACTOR": (0.05, 0.5),   # lowered floor: can suppress exploration harder
    "KILL_REWARD":           (2.0, 20.0),   # added: auto-loop can tune the primary lever
}

# writer.suggest speaks in lowercase knobs; map them onto the supervisor's env vars.
# (Exploration knobs COVERAGE/EXIT aren't in writer.suggest — the heuristic owns those.)
LLM_KNOB_TO_ENV = {
    "hit_reward": "HIT_REWARD",
    "miss_penalty": "MISS_PENALTY",
    "damage_taken_penalty": "DAMAGE_TAKEN_PENALTY",
    "death_penalty": "DEATH_PENALTY",
}


def propose(env: dict, m: dict, algo: str = "ppo") -> tuple[dict, str]:
    """Heuristic 'understanding -> action': nudge the knob that targets the weakest
    metric, within bounds. Returns (new_env, human-readable reason).

    `algo` selects the policy-exploration lever: PPO un-freezes via ENT_COEF (entropy),
    QR-DQN via DQN_EPS_FINAL (ε-greedy floor). Bumping the wrong one is a silent no-op."""
    new = dict(env)
    # The "un-freeze a collapsed policy" knob differs by algorithm.
    unfreeze_knob = "DQN_EPS_FINAL" if algo == "dqn" else "ENT_COEF"

    def bump(key, factor=None, add=None):
        lo, hi = BOUNDS[key]
        # Seed a missing knob from its lower bound so a *factor on an absent key isn't a no-op
        # (e.g. DQN_EPS_FINAL isn't in the seed env, so v would start at 0 → stays 0).
        v = float(new.get(key, lo))
        v = v * factor if factor is not None else v + add
        new[key] = round(max(lo, min(hi, v)), 4)

    timeout_rate = m.get("timeout_rate", 0.0)
    explored = m.get("explored_fraction", 0.0)
    # Rich-metric diagnosis (the panels): what's the reward ACTUALLY rewarding, is it spraying,
    # is it circling? These let the loop auto-make the fix we made by hand.
    rb = m.get("reward_breakdown", {}) or {}
    explore_share = rb.get("explore", 0.0)       # fraction of reward from exploration
    wasted = m.get("wasted_shot_rate", 0.0)      # shots fired with NO enemy on screen
    aim_off = m.get("aim_offset", 0.0)           # nearest enemy off-centre (1=edge)
    revisit = m.get("revisit_rate", 0.0)         # circling (revisited cells)

    # Timeout diagnosis: if > 80% of episodes time out AND exploration is low, the
    # episode is too short to let the agent find anything interesting — extend it.
    if timeout_rate > 0.80 and explored < 0.15:
        bump("EPISODE_TIMEOUT", factor=1.5)
        return new, (f"timeout_rate={timeout_rate:.0%}, explored={explored:.0%} → "
                     f"episode too short — raise EPISODE_TIMEOUT to {int(new['EPISODE_TIMEOUT'])}")

    if explored < 0.10 and explore_share < 0.6:
        # Very low exploration AND the reward isn't already explore-dominated: push on EVERY
        # exploration lever to break out of the spawn room. (If explore_share is already high,
        # pouring in MORE explore reward is the trap we hit by hand — fall through to the spray
        # rule below instead.)
        bump("COVERAGE_REWARD", factor=1.4)
        bump("FRONTIER_REWARD", factor=1.5)
        bump("RND_SCALE", factor=1.3)
        bump("GOEXPLORE_GOAL_PROB", add=0.1)
        return new, (f"explored only {explored:.0%} -> raise COVERAGE_REWARD to "
                     f"{new['COVERAGE_REWARD']}, FRONTIER_REWARD to {new['FRONTIER_REWARD']}, "
                     f"RND_SCALE to {new['RND_SCALE']}, GOEXPLORE_GOAL_PROB to "
                     f"{new['GOEXPLORE_GOAL_PROB']}")
    # Root-cause before paying it to explore: an agent that keeps DYING or whose policy is
    # FROZEN can never reach the exit, so fix survival/aliveness first.
    # High death rate: make damage + dying hurt more so it disengages at low HP (it now
    # knows its HEALTH via game_vars).
    if m.get("death_rate", 0.0) > 0.5:
        bump("DAMAGE_TAKEN_PENALTY", factor=1.3)
        bump("DEATH_PENALTY", factor=1.2)
        return new, (f"death_rate {m.get('death_rate',0):.0%} -> raise DAMAGE_TAKEN_PENALTY "
                     f"to {new['DAMAGE_TAKEN_PENALTY']}, DEATH_PENALTY to {new['DEATH_PENALTY']}")
    # SPRAY / reward-imbalance (rich metrics): it fires with no target (wasted_shot_rate high)
    # AND/OR the reward is dominated by EXPLORATION while it isn't aiming (aim_offset high) — the
    # exact miscalibration that makes it spray instead of aim. Cut the exploration pull, sharpen
    # the trigger + aim. This is the hand-made fix, automated.
    if wasted > 0.4 or (explore_share > 0.6 and aim_off > 0.5):
        bump("COVERAGE_REWARD", factor=0.5)
        bump("RND_SCALE", factor=0.5)
        bump("FRONTIER_REWARD", factor=0.5)
        bump("MISS_PENALTY", add=0.04)
        bump("ENGAGEMENT_REWARD", factor=1.4)
        return new, (f"spraying (wasted {wasted:.0%}, reward {explore_share:.0%} explore, "
                     f"aim_offset {aim_off:.2f}) -> cut exploration + sharpen trigger/aim: "
                     f"COVERAGE_REWARD {new['COVERAGE_REWARD']}, MISS_PENALTY {new['MISS_PENALTY']}, "
                     f"ENGAGEMENT_REWARD {new['ENGAGEMENT_REWARD']}")
    # Circling: covers little but revisits a lot — anti-circle levers (frontier + curiosity).
    if revisit > 0.85 and explored < 0.3 and explore_share < 0.6:
        bump("FRONTIER_REWARD", factor=1.5)
        bump("RND_SCALE", factor=1.3)
        return new, (f"circling (revisit {revisit:.0%}, explored {explored:.0%}) -> anti-circle: "
                     f"FRONTIER_REWARD {new['FRONTIER_REWARD']}, RND_SCALE {new['RND_SCALE']}")
    # COMBAT regime diagnosis (separate from exploration): the agent SEES enemies but won't
    # shoot (combat_engagement low) or barely kills -> the deterministic policy has frozen.
    # Un-freeze it (entropy) and pay it to face enemies. Uses the per-mode telemetry when
    # present (combat_fraction/engagement), else falls back to kills/ep.
    kills = m.get("kills_per_episode", 0.0)
    engagement = m.get("combat_engagement")
    saw_enemies = m.get("combat_fraction", 0.0) > 0.05
    combat_passive = (engagement is not None and saw_enemies and engagement < 0.3) \
        or (kills < 0.5 and timeout_rate < 0.6)
    if combat_passive:
        bump(unfreeze_knob, factor=1.3)
        bump("ENGAGEMENT_REWARD", factor=1.5)
        why = (f"combat_engagement={engagement:.0%} (sees enemies, won't shoot)"
               if engagement is not None and saw_enemies else f"kills/ep={kills:.2f}")
        return new, (f"passive in combat ({why}) -> un-freeze policy: {unfreeze_knob} to "
                     f"{new[unfreeze_knob]}, ENGAGEMENT_REWARD to {new['ENGAGEMENT_REWARD']}")
    if explored < 0.5:
        bump("COVERAGE_REWARD", factor=1.3)
        return new, f"explored only {explored:.0%} -> raise COVERAGE_REWARD to {new['COVERAGE_REWARD']}"
    if m.get("shooting_accuracy", 0.0) < 0.10:
        bump("MISS_PENALTY", add=0.05)
        bump("HIT_REWARD", factor=1.2)
        return new, f"accuracy {m.get('shooting_accuracy',0):.0%} -> MISS_PENALTY {new['MISS_PENALTY']}, HIT_REWARD {new['HIT_REWARD']}"
    # Last resort once survival/exploration/aim are healthy: nudge the exit reward itself.
    if m.get("exit_rate", 0.0) == 0.0:
        bump("EXIT_REWARD", factor=1.3)
        bump("COVERAGE_REWARD", factor=1.2)  # exploring helps find the exit
        return new, f"never reached the exit -> raise EXIT_REWARD to {new['EXIT_REWARD']}, COVERAGE to {new['COVERAGE_REWARD']}"
    # Everything healthy: anneal exploration bonus to consolidate the policy.
    bump("COVERAGE_REWARD", factor=0.8)
    return new, f"metrics healthy -> anneal COVERAGE_REWARD to {new['COVERAGE_REWARD']}"


def llm_propose(cfg: Config, env: dict, m: dict) -> Optional[tuple[dict, str]]:
    """LLM-driven proposal for the COMBAT knobs (hit/miss/damage/death), reusing the
    same offline reward-suggestions model as `writer.suggest`. It's grounded in the
    cross-run event memory plus this iteration's measured accuracy. Each suggestion is
    clamped to BOUNDS (the guardrail still applies). Returns (new_env, reason), or None
    if the LLM is unavailable / has nothing usable so the caller keeps the heuristic.

    Exploration knobs (COVERAGE_REWARD/EXIT_REWARD) are intentionally NOT touched here —
    they're the GOAL's top weight and live outside writer.suggest, so the heuristic
    owns them and the LLM only refines combat on top."""
    try:
        from writer.llm_client import LLMWriter
        from writer.memory_store import MemoryStore
        from writer.reflect import aggregate_events

        stats = aggregate_events(MemoryStore.read_events(cfg.memory_dir))
        if stats["total"] < cfg.min_events_for_lessons:
            print(f"[autonomous] only {stats['total']} event(s) — LLM proposer holds off.")
            return None
        stats["shooting_accuracy"] = float(m.get("shooting_accuracy", 0.0))
        weights = cfg.reward_weights()
        llm = LLMWriter(model=cfg.llm_model, host=cfg.ollama_host,
                        num_ctx=cfg.llm_num_ctx, num_predict=cfg.llm_num_predict,
                        keep_alive=cfg.llm_keep_alive)
        res = llm.generate_reward_suggestions(stats, weights)
    except Exception as e:
        print(f"[autonomous] LLM proposer unavailable ({e}); using heuristic.")
        return None

    new = dict(env)
    applied = []
    for t in res.tweaks:
        envk = LLM_KNOB_TO_ENV.get(t.knob)
        if not envk or envk not in BOUNDS:
            continue
        lo, hi = BOUNDS[envk]
        clamped = round(max(lo, min(hi, float(t.suggested))), 4)
        if clamped != float(new.get(envk, 0.0)):
            new[envk] = clamped
            applied.append(f"{envk}->{clamped}")
    if not applied:
        return None
    return new, f"LLM: {res.summary.strip()[:120]} ({', '.join(applied)})"


def situation_text(m: dict) -> str:
    """One-line description of the agent's CURRENT behaviour — what the semantic memory embeds, so
    'similar past situations' are matched by MEANING (not keyword)."""
    rb = m.get("reward_breakdown", {}) or {}
    parts = [
        f"explored {m.get('explored_fraction', 0):.0%}",
        f"exit_progress {m.get('exit_progress', 0):.0%}",
        f"kills {m.get('kills_per_episode', 0):.1f}",
        f"accuracy {m.get('shooting_accuracy', 0):.0%}",
        f"wasted_shots {m.get('wasted_shot_rate', 0):.0%}",
        f"aim_offset {m.get('aim_offset', 0):.2f}",
        f"revisit {m.get('revisit_rate', 0):.0%}",
        f"deaths {m.get('death_rate', 0):.0%}",
        f"reward_explore {rb.get('explore', 0):.0%}",
        f"reward_combat {rb.get('combat', 0):.0%}",
    ]
    return "agent behaviour: " + ", ".join(parts)


def semantic_recall(memory_dir: str, m: dict, top_k: int = 3) -> tuple[Optional[dict], str]:
    """Ask semantic memory: 'have I seen a situation like THIS, and what change worked?' Returns
    (env_delta, note) from the most-similar KEPT iteration that improved, or (None, '')."""
    try:
        from writer.semantic_memory import SemanticMemory
        sm = SemanticMemory(memory_dir)
        hits = sm.search(situation_text(m), top_k=top_k)
        sm.close()
    except Exception:
        return None, ""
    for text, meta, score in hits or []:
        meta = meta or {}
        chg = meta.get("change")
        if chg and meta.get("kept") and float(meta.get("score", 0)) > 0 and score >= 0.6:
            return dict(chg), (f"semantic recall ({score:.2f} similar past run "
                               f"scored {float(meta.get('score',0)):.2f}): {str(meta.get('reason',''))[:70]}")
    return None, ""


def semantic_record(memory_dir: str, m: dict, change: dict, score: float, kept: bool,
                    reason: str = "") -> None:
    """Store this iteration's (situation → change → outcome) so a future similar situation recalls
    what worked. Best-effort (no-op if semantic memory is unavailable)."""
    try:
        from writer.semantic_memory import SemanticMemory
        sm = SemanticMemory(memory_dir)
        sm.add(situation_text(m), meta={"change": dict(change or {}), "score": float(score),
                                        "kept": bool(kept), "reason": str(reason)[:160]})
        sm.close()
    except Exception:
        pass


def llm_propose_open(cfg: Config, env: dict, m: dict) -> Optional[tuple[dict, str]]:
    """OPEN LLM proposer: hand the model the FULL parameter catalog (every tunable knob, its
    current value, range and effect) + this run's metrics, and let it propose a new value for
    ANY of them. Every change is validated/clamped against the registry, so it can ask for
    anything but only valid, in-range params apply. Returns (new_env, reason) or None.

    This is the "the LLM knows and can change all parameters" path (vs llm_propose's fixed
    combat subset). Needs Ollama; degrades to None if unavailable."""
    try:
        from pydantic import BaseModel
        from rl.tuning_registry import describe_for_llm, validate
        from writer.llm_client import LLMWriter

        class _Change(BaseModel):
            param: str
            value: float

        class _Proposal(BaseModel):
            changes: list[_Change]
            reason: str

        catalog = describe_for_llm(env)
        keymetrics = {k: m.get(k) for k in (
            "aim_offset", "wasted_shot_rate", "kill_conversion", "shooting_accuracy",
            "explored_fraction", "revisit_rate", "exit_progress", "death_rate",
            "kills_per_episode", "reward_breakdown") if k in m}
        system = ("You tune a Doom RL agent's reward + training parameters. Given the metrics and "
                  "the full parameter catalog, propose new values for the few knobs most likely to "
                  "fix the weakest behaviour. Change only what helps; stay within each range.")
        user = f"METRICS:\n{keymetrics}\n\n{catalog}\n\nReturn the changes and a one-line reason."
        llm = LLMWriter(model=cfg.llm_model, host=cfg.ollama_host,
                        num_ctx=cfg.llm_num_ctx, num_predict=cfg.llm_num_predict,
                        keep_alive=cfg.llm_keep_alive)
        content = llm._chat(system, user, _Proposal.model_json_schema())
        prop = _Proposal.model_validate_json(content)
    except Exception as e:
        print(f"[autonomous] open LLM proposer unavailable ({e}); skipping.")
        return None
    proposal = {c.param: c.value for c in prop.changes}
    new = validate(proposal, base_env=env)
    applied = [f"{k}->{new[k]}" for k in proposal if k in new and new.get(k) != env.get(k)]
    if not applied:
        return None
    return new, f"LLM(open): {prop.reason.strip()[:120]} ({', '.join(applied)})"


def propose_next(cfg: Config, env: dict, m: dict, use_llm: bool,
                 algo: str = "ppo") -> tuple[dict, str]:
    """Pick the next reward config. The heuristic always runs (it owns exploration and
    is the fallback); the persistent MEMORY refines combat on top (targets the real death
    mode across all runs, and never repeats a change a past experiment disproved); when
    --llm is on, the LLM refines combat too."""
    new, reason = propose(env, m, algo=algo)
    # Memory-informed: draw on the whole persistent history, not just this iteration's eval.
    try:
        from writer.memory_policy import recall_proposal
        mem_env, mem_reason = recall_proposal(cfg.memory_dir, new)
        if mem_env is not None:
            new, reason = mem_env, f"{reason}; {mem_reason}"
    except Exception as e:
        print(f"[autonomous] memory policy unavailable ({type(e).__name__}); skipping.")
    if use_llm:
        llm_res = llm_propose(cfg, new, m)
        if llm_res:
            new, llm_reason = llm_res
            reason = f"{reason}; {llm_reason}"
    return new, reason


def _subprocess_env(env: dict) -> dict:
    """subprocess requires str env values, but propose() puts numeric reward weights
    (floats) into the dict — coerce everything to str so a resume after a tweak doesn't
    crash with 'expected str ... not float'."""
    return {**os.environ, **{k: str(v) for k, v in env.items()}}


def train_chunk(env: dict, doom_map: str, steps: int, fresh: bool,
                algo: str = "ppo") -> None:
    n_envs = str(env.get("N_ENVS") or os.getenv("N_ENVS", "8"))
    if algo == "dqn":
        # train_dqn resumes the latest checkpoint by DEFAULT (no --resume flag exists).
        # Only --fresh is meaningful. n-envs is inherited from N_ENVS in the subprocess
        # env, but we pass it explicitly so the log shows the real value.
        cmd = [PY, "-m", "rl.train_dqn",
               "--map", doom_map, "--timesteps", str(steps), "--n-envs", n_envs]
        if fresh:
            cmd.append("--fresh")
    else:
        cmd = [PY, "-m", "rl.train", "--maps", doom_map,
               "--n-envs", n_envs, "--timesteps", str(steps)]
        cmd.append("--fresh" if fresh else "--resume")
    subprocess.run(cmd, env=_subprocess_env(env), check=True)


def eval_brain(env: dict, episodes: int, temperature: Optional[float] = None,
               algo: str = "ppo") -> dict:
    cmd = [PY, "-m", "rl.eval", "--episodes", str(episodes), "--json", "--algo", algo]
    # Score the TEMPERED policy, not the raw argmax: this agent's argmax collapses to a
    # passive action while the learned distribution explores+fights. Scoring argmax would
    # make the supervisor optimise a frozen policy. T (e.g. 0.5) measures real capability.
    # (QR-DQN ignores temperature — it's value-based; the loop passes None for dqn.)
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    out = subprocess.run(
        cmd, env=_subprocess_env(env), check=True, capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("METRICS_JSON "):
            return json.loads(line[len("METRICS_JSON "):])
    raise RuntimeError("eval produced no METRICS_JSON")


def load_history(cfg: Config) -> list:
    """Restore a prior auto session's trail from autonomy.jsonl (for --resume). Returns []
    if nothing is stored yet."""
    path = os.path.join(cfg.memory_dir, "autonomy.jsonl")
    if not os.path.exists(path):
        return []
    history = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return history


def _refresh_db(cfg: Config) -> None:
    """Re-sync the SQLite read-view from the JSONL stores. Best-effort: a failure here
    must never abort a training run (the JSONL remains the source of truth)."""
    try:
        from writer import db
        db.build(cfg.memory_dir)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[autonomous] db refresh skipped: {exc}")


def _record_iteration(cfg: Config, i: int, prev_env: dict, eval_env: dict,
                      kept: bool, sc: float) -> None:
    """Auto-chain (P4): record this iteration's reward change into the experiment registry —
    the full trail INCLUDING reversions. Uses result 'kept'/'reverted' (NOT 'improved') on
    purpose: single-seed auto decisions populate the registry and rollback history, but are
    NOT auto-adopted into learned_config (that stays the job of the multi-seed `experiment`
    command) and don't pollute the 'validated' knowledge tier. Best-effort."""
    if i == 0:
        return
    from writer.rollback import RollbackLog, diff_envs
    before, change, after = diff_envs(prev_env, eval_env)
    if not change:
        return
    try:
        # Structured rollback log (the audit trail: before/change/after/result/kept).
        RollbackLog(cfg.memory_dir).record(i, before, change, after,
                                           {"score": round(float(sc), 4)}, kept)
        # Mirror into the SQLite experiment registry (queryable view). 'kept'/'reverted'
        # NOT 'improved' -> single-seed auto decisions are logged but never auto-adopted.
        from writer.db import insert_experiment
        desc = "; ".join(f"{k} {o}→{n}" for k, (o, n) in change.items())
        insert_experiment(cfg.memory_dir, param=f"auto iter {i}: {desc}",
                          old_val="", new_val="", result="kept" if kept else "reverted",
                          confidence=0.3, notes=f"score={sc:.3f}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[autonomous] rollback/registry log skipped: {exc}")


def write_log(cfg: Config, history: list) -> None:
    """Persist the self-improvement trail: JSONL (machine) + live Obsidian log (human)."""
    os.makedirs(cfg.memory_dir, exist_ok=True)
    with open(os.path.join(cfg.memory_dir, "autonomy.jsonl"), "w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")

    note = os.path.join(cfg.vault_path, cfg.dir_index, "Autonomy Log.md")
    os.makedirs(os.path.dirname(note), exist_ok=True)
    best = max(history, key=lambda h: h["score"])
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Score trajectory (sparkline text)
    scores = [round(h["score"], 2) for h in history]
    score_trend = " → ".join(
        f"{'**' if s == max(scores) else ''}{s}{'**' if s == max(scores) else ''}"
        for s in scores
    )

    lines = [
        "---", "type: autonomy-log",
        f"updated: {ts}",
        f"iterations: {len(history)}",
        f"best_score: {best['score']:.3f}",
        "tags: [autonomy, doom-rl]", "---", "",
        "# Autonomy Log — agent improving itself",
        "",
        "> Each iteration: train → eval → score → propose reward delta → keep if improved.",
        "> Score = 4×exit_rate + 3×exploration + 0.5×kills + 1×accuracy",
        "",
        f"**Score trajectory:** {score_trend}",
        "",
        "## Iteration table",
        "",
        "> **Combat** = of the time it SEES an enemy, how often it actually shoots "
        "(low = passive). **Explored/Exit** = the exploration regime. The two are tuned "
        "separately by the coach.",
        "",
        "| Iter | Explored | Exit% | Kills/ep | Acc | Combat | Score | Δ | Kept? | Decision |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    prev_score = None
    for h in history:
        m = h["metrics"]
        delta = f"{h['score'] - prev_score:+.2f}" if prev_score is not None else "—"
        prev_score = h["score"]
        kept_icon = "✅" if h["kept"] else "↩ reverted"
        # Combat regime (per-mode telemetry): blank for older records that predate it.
        combat = (f"{m['combat_engagement']:.0%}" if m.get("combat_engagement") is not None
                  and m.get("combat_fraction", 0.0) > 0.0 else "—")
        lines.append(
            f"| {h['iter']} "
            f"| {m['explored_fraction']:.0%} "
            f"| {m['exit_rate']:.0%} "
            f"| {m['kills_per_episode']:.2f} "
            f"| {m['shooting_accuracy']:.0%} "
            f"| {combat} "
            f"| **{h['score']:.2f}** "
            f"| {delta} "
            f"| {kept_icon} "
            f"| {h['reason']} |"
        )

    # Reward deltas — what actually changed per iteration. Index by LIST POSITION, not by
    # h["iter"]: a failed iteration is skipped (not appended), so iter numbers can have gaps
    # and `history[iter-1]` would index out of range (and compare the wrong pair).
    lines += ["", "## Reward changes applied", ""]
    for idx in range(1, len(history)):
        h = history[idx]
        prev_env = history[idx - 1]["env"]
        curr_env = h["env"]
        changed = {k: (prev_env.get(k), curr_env.get(k))
                   for k in set(prev_env) | set(curr_env)
                   if prev_env.get(k) != curr_env.get(k) and k in BOUNDS}
        if changed:
            changes_str = "  ".join(f"`{k}`: {old}→{new}"
                                    for k, (old, new) in changed.items())
            kept = "✅" if h["kept"] else "↩"
            lines.append(f"- Iter {h['iter']} {kept}: {changes_str}")

    lines += [
        "",
        f"## Best configuration (iter {best['iter']}, score **{best['score']:.2f}**)",
        "",
        "Apply to `.env` to lock in the agent's self-discovered best setup:",
        "",
        "```bash",
        *[f"{k}={best['env'][k]}" for k in BOUNDS if k in best["env"]],
        "```",
        "",
        "---",
        f"_Updated at {ts} · [[Knowledge Graph]]_",
    ]
    with open(note, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_final_report(
    cfg: Config,
    history: list,
    doom_map: str,
    use_llm: bool = False,
) -> str:
    """Write a comprehensive final report to 30-runs/ after all iterations complete.

    Covers: session narrative, before/after performance, what was tried,
    what worked, best config, behavior flags, and (if --llm) an LLM synthesis.
    Returns the path of the note written.
    """
    from writer.memory_store import MemoryStore
    from writer.behavior import detect, write_behavior_note
    from writer.snapshot_log import SnapshotLog, log_path_for

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    best = max(history, key=lambda h: h["score"])
    first = history[0]
    # Compare each entry to the PREVIOUS list element (not history[iter-1]) — failed
    # iterations are skipped, so iter numbers have gaps that would misindex.
    improved = [history[i] for i in range(1, len(history))
                if history[i]["kept"] and history[i]["score"] > history[i - 1]["score"]]
    reverted = [h for h in history if not h["kept"]]

    m0 = first["metrics"]
    mb = best["metrics"]

    # Behavior flags from the vault memory
    events = MemoryStore.read_events(cfg.memory_dir)
    snap_path = log_path_for(cfg.pending_dir, cfg.run_name)
    snaps = SnapshotLog.read_all(snap_path)
    flags = detect(events, snaps)
    if flags:
        write_behavior_note(cfg, flags)

    # LLM narrative (optional)
    llm_narrative = ""
    if use_llm and events:
        try:
            from writer.llm_client import LLMWriter
            from writer.reflect import aggregate_events
            stats = aggregate_events(events)
            llm = LLMWriter(model=cfg.llm_model, host=cfg.ollama_host,
                            num_ctx=cfg.llm_num_ctx, num_predict=cfg.llm_num_predict,
                            keep_alive=cfg.llm_keep_alive)
            # Build a prompt from the autonomy session
            session_summary = (
                f"Autonomy session on {doom_map}: {len(history)} iterations, "
                f"{len(improved)} improvements, {len(reverted)} reverted. "
                f"Score {first['score']:.2f} → {best['score']:.2f} "
                f"(+{best['score'] - first['score']:.2f}). "
                f"Exploration {m0['explored_fraction']:.0%} → {mb['explored_fraction']:.0%}. "
                f"Kills {m0['kills_per_episode']:.1f} → {mb['kills_per_episode']:.1f}/ep. "
                f"Exit rate {m0['exit_rate']:.0%} → {mb['exit_rate']:.0%}."
            )
            note_obj = llm.generate_lessons({
                **stats,
                "session": session_summary,
                "total": max(stats["total"], 1),
            })
            if note_obj and note_obj.lessons:
                llm_narrative = "\n".join(
                    f"### {l.title}\n\n{l.insight}\n\n_Evidence: {l.evidence}_\n"
                    for l in note_obj.lessons
                )
        except Exception as e:
            llm_narrative = f"_(LLM synthesis failed: {e})_"

    # Build the report
    out_dir = os.path.join(cfg.vault_path, cfg.dir_runs)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"Auto Session — {doom_map} — {ts[:10]}.md")

    score_improvement = best["score"] - first["score"]
    sign = "+" if score_improvement >= 0 else ""

    lines = [
        "---",
        "type: auto-session-report",
        f"created: {ts}",
        f"map: {doom_map}",
        f"iterations: {len(history)}",
        f"score_start: {first['score']:.3f}",
        f"score_best: {best['score']:.3f}",
        f"score_delta: {score_improvement:.3f}",
        "tags: [autonomy, session-report, doom-rl]",
        "---",
        "",
        f"# Auto Session — {doom_map} — {ts[:10]}",
        "",
        f"> **{len(history)} iterations** · Score {first['score']:.2f} → **{best['score']:.2f}** "
        f"({sign}{score_improvement:.2f}) · "
        f"{len(improved)} improvement(s) · {len(reverted)} revert(s)",
        "",
        "## Performance: before → after",
        "",
        "| Metric | Start (iter 0) | Best (iter {}) | Delta |".format(best["iter"]),
        "|--------|----------------|----------------|-------|",
        f"| Exploration | {m0['explored_fraction']:.0%} | **{mb['explored_fraction']:.0%}** "
        f"| {mb['explored_fraction'] - m0['explored_fraction']:+.0%} |",
        f"| Exit rate   | {m0['exit_rate']:.0%} | **{mb['exit_rate']:.0%}** "
        f"| {mb['exit_rate'] - m0['exit_rate']:+.0%} |",
        f"| Kills/ep    | {m0['kills_per_episode']:.2f} | **{mb['kills_per_episode']:.2f}** "
        f"| {mb['kills_per_episode'] - m0['kills_per_episode']:+.2f} |",
        f"| Accuracy    | {m0['shooting_accuracy']:.0%} | **{mb['shooting_accuracy']:.0%}** "
        f"| {mb['shooting_accuracy'] - m0['shooting_accuracy']:+.0%} |",
        f"| Score       | {first['score']:.2f} | **{best['score']:.2f}** "
        f"| {sign}{score_improvement:.2f} |",
        "",
        "## Iteration-by-iteration",
        "",
        "| # | Score | Δ | Decision | Kept? |",
        "|---|-------|---|----------|-------|",
    ]
    prev = None
    for h in history:
        delta = f"{h['score'] - prev:+.2f}" if prev is not None else "baseline"
        prev = h["score"]
        lines.append(
            f"| {h['iter']} | {h['score']:.2f} | {delta} | {h['reason']} "
            f"| {'✅' if h['kept'] else '↩ reverted'} |"
        )

    # What actually changed
    lines += ["", "## Reward adjustments tried", ""]
    any_change = False
    for idx in range(1, len(history)):
        h = history[idx]
        prev_env = history[idx - 1]["env"]
        curr_env = h["env"]
        changed = {k: (prev_env.get(k), curr_env.get(k))
                   for k in BOUNDS
                   if prev_env.get(k) != curr_env.get(k)}
        if changed:
            any_change = True
            kept = "✅ kept" if h["kept"] else "↩ reverted"
            lines.append(f"**Iter {h['iter']} ({kept}):**")
            for k, (old, new) in changed.items():
                lines.append(f"- `{k}`: {old} → {new}")
            lines.append("")
    if not any_change:
        lines.append("_No reward changes were tried (all iterations used the same config)._")
        lines.append("")

    # Best config
    lines += [
        "## Best config to apply",
        "",
        "_Copy these into `.env` and run `doom-cli train --map {doom_map} --resume`:_".format(
            doom_map=doom_map),
        "",
        "```bash",
        *[f"{k}={best['env'][k]}" for k in BOUNDS if k in best["env"]],
        "```",
        "",
    ]

    # Behavior flags
    if flags:
        lines += ["## Behavior flags detected", ""]
        for f in sorted(flags, key=lambda x: -x.confidence):
            icon = "🔴" if f.confidence >= 0.7 else "🟡"
            lines.append(f"- {icon} **{f.name}** ({f.confidence:.0%}): {f.description}")
            lines.append(f"  → {f.recommendation}")
        lines.append("")
    else:
        lines += ["## Behavior flags", "", "_No flags detected._", ""]

    # LLM narrative
    if llm_narrative:
        lines += ["## LLM synthesis", "", llm_narrative]
    elif use_llm:
        lines += ["## LLM synthesis", "", "_(LLM not available or no events yet.)_", ""]

    # Links
    lines += [
        "---",
        "",
        "## Links",
        "",
        "- [[Autonomy Log]] (live iteration log)",
        f"- [[Map - {doom_map}]]",
        "- [[Knowledge Graph]]",
    ]
    if flags:
        lines.append("- [[Behavior]] (80-recommendations)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[autonomous] final report → {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous self-improving supervisor.")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--steps", type=int, default=100000, help="Timesteps per iteration.")
    p.add_argument("--map", default=None, help="Map to train on (default: cfg.maps[0]).")
    p.add_argument("--episodes", type=int, default=10, help="Eval episodes per iteration.")
    # Resume is the DEFAULT (continue the brain + prior session). --fresh/--clear restarts.
    p.add_argument("--fresh", "--clear", dest="fresh", action="store_true",
                   help="Start a brand-new session from zero (fresh brain + cleared history). "
                        "Default behaviour (no flag) is to RESUME the existing brain.")
    p.add_argument("--spatial", action="store_true",
                   help="Enable spatial memory (2nd obs channel). Forces --fresh (obs shape changes).")
    p.add_argument("--rnd", action="store_true",
                   help="Enable RND intrinsic curiosity (position-based novelty bonus).")
    p.add_argument("--goexplore", action="store_true",
                   help="Enable Go-Explore frontier-goal resets (return-then-explore).")
    p.add_argument("--depth", action="store_true",
                   help="Enable depth-perception obs channel (ViZDoom depth buffer). Forces --fresh.")
    p.add_argument("--strafe", action="store_true",
                   help="Add strafe (sideways) actions. Changes the action count → forces --fresh.")
    p.add_argument("--automap", action="store_true",
                   help="Enable the native top-down automap obs channel. Forces --fresh.")
    p.add_argument("--game-vars", dest="game_vars", action="store_true",
                   help="Feed HEALTH/AMMO into the policy (the agent knows its own state).")
    p.add_argument("--llm", action="store_true",
                   help="Let the offline reward-suggestions LLM refine the combat knobs "
                        "(needs Ollama + enough events); falls back to the heuristic.")
    p.add_argument("--graph", action="store_true",
                   help="Use the LangGraph coach (V2 Phase 4): explicit observe→diagnose→"
                        "hypothesize→propose→validate graph instead of the heuristic cascade.")
    p.add_argument("--resume", action="store_true",
                   help="(default behaviour — kept for back-compat) Continue the prior session.")
    p.add_argument("--fast", action="store_true",
                   help="Throughput mode WITHOUT disabling any perception: scale the parallel "
                        "envs to your CPU cores (ViZDoom is CPU-bound, so this is the free "
                        "speedup). Everything the agent sees stays on.")
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"],
                   help="RL algorithm: ppo (default, on-policy) or dqn (QR-DQN, off-policy "
                        "with replay buffer — more sample-efficient, V2 default).")
    args = p.parse_args()

    cfg = Config()
    doom_map = args.map or cfg.maps[0]

    # --fast: use the machine's cores for more parallel envs (pure throughput, nothing turned
    # off). Cap at 8 to stay within ~16GB RAM with all perception channels on; leave 2 cores
    # for the OS / the policy update. Honest about what it changed.
    if args.fast:
        import os as _os
        cores = _os.cpu_count() or 4
        cfg.n_envs = max(cfg.n_envs, min(8, max(2, cores - 2)))
        print(f"[autonomous] --fast: {cores} cores detected -> N_ENVS={cfg.n_envs} "
              f"(all perception channels stay ON).")

    # --fresh/--clear means a true restart: wipe the prior session trail so it doesn't get
    # resumed. (The brain itself is overwritten by the first --fresh training chunk.)
    if args.fresh:
        _trail = os.path.join(cfg.memory_dir, "autonomy.jsonl")
        if os.path.exists(_trail):
            os.remove(_trail)
            print("[autonomous] --fresh/--clear: cleared prior session history.")

    spatial = args.spatial or cfg.spatial_memory
    use_rnd  = args.rnd    or cfg.use_rnd
    depth    = args.depth  or cfg.depth_perception
    strafe   = args.strafe or cfg.strafe
    automap  = args.automap or cfg.automap
    gamevars = args.game_vars or cfg.game_vars
    # Write the resolved perception flags back to cfg so brain_prefix / _dqn_prefix /
    # build_vec_env all derive the SAME brain name (the DQN prefix reads them off cfg).
    cfg.game_vars = gamevars
    cfg.spatial_memory = spatial
    cfg.depth_perception = depth
    cfg.automap = automap
    cfg.strafe = strafe

    # NOTE: perception/action flags change the obs shape / action count, but the brain NAME
    # now ENCODES all of them (e.g. `..._a15_sp`), so the lookup below finds the exact
    # compatible brain — no blanket force-fresh needed (an unconditional one was a bug that
    # discarded a resumable brain). Resume is the DEFAULT; only --fresh/--clear starts over.

    algo = getattr(args, "algo", "ppo")

    # Auto-fresh ONLY when no compatible brain exists yet (a truly new family) — otherwise
    # continue training the existing brain. Derives the exact brain family name from the
    # current flags (action count, lstm, spatial/depth/automap, frame_stack).
    if not args.fresh:
        import glob as _glob
        from doom.campaign import campaign_metadata
        from rl.algo import brain_prefix
        meta = campaign_metadata(cfg.wad_path, doom_map, strafe=strafe)
        if algo == "dqn":
            from rl.train_dqn import _dqn_prefix
            name_prefix = _dqn_prefix(meta["num_actions"], cfg.game_vars, cfg)
        else:
            name_prefix = brain_prefix("campaign", meta["num_actions"], cfg.use_lstm,
                                       spatial, depth, automap, cfg.frame_stack, cfg.game_vars,
                                       getattr(cfg, "semantic_channel", False))
        ckpt_dir = cfg.checkpoint_dir
        has_brain = bool(
            os.path.exists(os.path.join(ckpt_dir, f"{name_prefix}_final.zip"))
            or _glob.glob(os.path.join(ckpt_dir, f"{name_prefix}_*_steps.zip"))
        )
        if not has_brain:
            print("[autonomous] no brain found in vault — auto-switching to --fresh")
            args.fresh = True

    # Seed the evolving reward env from the current config (campaign mode, no docs:
    # the supervisor is fast; documentation is a separate, final concern).
    env = {
        "CAMPAIGN": "1", "MAPS": doom_map, "DOCS_ENABLED": "0", "MEMORY_ENABLED": "1",
        "CONTROL_ENABLED": "0", "N_ENVS": str(cfg.n_envs),
        # PPO/obs hyperparams pinned so the train AND eval subprocesses agree (the brain
        # name encodes frame_stack, so a mismatch would look for the wrong checkpoint).
        "FRAME_STACK": str(cfg.frame_stack), "ENT_COEF": str(cfg.ent_coef),
        "GAME_VARS": "1" if gamevars else "0",
        "USE_LABELS": "1" if cfg.use_labels else "0",
        "ENGAGEMENT_REWARD": str(cfg.engagement_reward),
        "SPATIAL_MEMORY": "1" if spatial else "0",
        "DEPTH_PERCEPTION": "1" if depth else "0",
        "STRAFE": "1" if strafe else "0",
        "AUTOMAP": "1" if automap else "0",
        "USE_LSTM": "1" if cfg.use_lstm else "0",
        "USE_RND": "1" if use_rnd else "0",
        "RND_SCALE": str(cfg.rnd_scale),
        "GOEXPLORE_GOAL_PROB": str(cfg.goexplore_goal_prob if not args.goexplore else max(cfg.goexplore_goal_prob, 0.3)),
        "GOEXPLORE_GOAL_SCALE": str(cfg.goexplore_goal_scale),
        "GOEXPLORE_REACH_RADIUS": str(cfg.goexplore_reach_radius),
        "FRONTIER_REWARD": str(cfg.frontier_reward),
        "EPISODE_TIMEOUT": str(cfg.episode_timeout),
        "COVERAGE_REWARD": str(cfg.coverage_reward), "EXIT_REWARD": str(cfg.exit_reward),
        "HIT_REWARD": str(cfg.hit_reward), "MISS_PENALTY": str(cfg.miss_penalty),
        "DAMAGE_TAKEN_PENALTY": str(cfg.damage_taken_penalty),
        "DEATH_PENALTY": str(cfg.death_penalty), "MOVE_REWARD": str(cfg.move_reward),
        "LIVING_REWARD": str(cfg.living_reward),
        "COMBAT_EXPLORE_SPLIT": "1" if cfg.combat_explore_split else "0",
        "COMBAT_EXPLORE_FACTOR": str(cfg.combat_explore_factor),
        "AUTO_USE": "1" if cfg.auto_use else "0",
        "DISCOVERY_REWARD": str(cfg.discovery_reward),
    }

    # Accumulate across sessions: overlay reward knobs the agent has PROVEN help (validated
    # experiments), then adopt any "improved" verdicts sitting in memory from prior runs.
    try:
        from writer.learned_config import LearnedConfig
        from writer.memory_policy import adopt_improved_experiments
        adopt_improved_experiments(cfg.memory_dir)
        learned = LearnedConfig(cfg.memory_dir).values()
        if learned:
            env = LearnedConfig(cfg.memory_dir).apply_to_env(env)
            print(f"[autonomous] applied learned config (proven knobs): {learned}")
    except Exception as e:
        print(f"[autonomous] learned config unavailable ({type(e).__name__}); skipping.")

    # LangGraph coach (V2 Phase 4): explicit observe→diagnose→hypothesize→propose→validate graph.
    # Falls back to the legacy heuristic if --graph is not passed or import fails.
    _coach = None
    if getattr(args, "graph", False):
        try:
            from rl.coach_graph import CoachGraph
            _coach = CoachGraph(cfg, use_llm=args.llm, algo=algo)
            print("[autonomous] using LangGraph coach (graph mode)")
        except Exception as exc:
            print(f"[autonomous] LangGraph coach unavailable ({exc}); using heuristic.")

    history = []
    best_score = -1e9
    start_iter = 0
    # Resume is the DEFAULT: restore a prior session's trail and continue, so a long auto run
    # survives a kill/restart. Only --fresh/--clear starts a brand-new session from zero.
    if not args.fresh:
        history = load_history(cfg)
        if history:
            best_score = max(h["score"] for h in history)
            env = history[-1].get("_next_env", env)  # the config queued for the next iter
            start_iter = len(history)
            print(f"[autonomous] --resume: restored {start_iter} prior iters "
                  f"(best score {best_score:.2f}); continuing from iter {start_iter}.")

    # --iterations is the number of NEW iterations to run THIS session (not a global total).
    # So resuming a session that already has N iters and asking for 8 runs 8 MORE — otherwise
    # `range(start_iter, iterations)` would be empty once start_iter caught up (did nothing).
    end_iter = start_iter + args.iterations
    # QR-DQN is value-based → scored by deterministic argmax (temperature can't apply).
    scoring = ("deterministic argmax" if algo == "dqn"
               else (f"tempered T={cfg.eval_temperature}" if cfg.eval_temperature > 0
                     else "argmax"))
    print(f"[autonomous] {args.iterations} new iterations (iters {start_iter}→{end_iter - 1}) "
          f"× {args.steps} steps on {doom_map} "
          f"({'LLM-refined' if args.llm else 'heuristic'} reward proposals; "
          f"algo={algo}; scoring={scoring})")
    for i in range(start_iter, end_iter):
        fresh = args.fresh and i == 0
        # Guard history[-1]: if iter 0 itself FAILED (e.g. a crashed eval), history is still
        # empty when i advances — accessing history[-1] then would abort the whole loop with
        # an IndexError (the opposite of the graceful-continue we want).
        reason = ("baseline" if i == 0 or not history
                  else history[-1].get("_next_reason", "adjust"))
        print(f"\n===== ITER {i} ({'fresh' if fresh else 'resume'}) — {reason} =====")
        # A single iteration crash (ViZDoom hiccup, OOM, transient subprocess failure)
        # must NOT abort the whole self-improvement run — that's the opposite of autonomy.
        # Catch it, keep the best brain found so far, and continue / finish gracefully.
        try:
            train_chunk(env, doom_map, args.steps, fresh, algo=algo)
            # Score the tempered policy (PPO). QR-DQN is value-based (argmax) — temperature
            # can't apply, so score it deterministically (its honest measure).
            temp = (None if algo == "dqn"
                    else (cfg.eval_temperature if cfg.eval_temperature > 0 else None))
            m = eval_brain(env, args.episodes, temperature=temp, algo=algo)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[autonomous] iter {i} FAILED ({type(e).__name__}): {e}")
            print("[autonomous] keeping best-so-far and continuing with the last good config.")
            if history:
                env = history[-1]["env"]  # fall back to the last config that ran
            continue

        sc = score(m)
        kept = (i == 0) or (sc >= best_score - 0.05)  # guardrail: revert regressions
        eval_env = dict(env)  # the config ACTUALLY evaluated this iter (before any rollback)
        print(f"[autonomous] iter {i}: score={sc:.2f} (best={best_score:.2f}) "
              f"explored={m['explored_fraction']:.0%} exit={m['exit_rate']:.0%} "
              f"kills={m['kills_per_episode']:.2f} -> {'KEEP' if kept else 'REVERT'}")

        # Auto-chain (P4): log this change + its verdict into the experiment registry.
        _record_iteration(cfg, i, history[-1]["env"] if history else {}, eval_env, kept, sc)

        if not kept:
            env = history[-1]["env"]  # roll back to the last good reward config
        else:
            best_score = max(best_score, sc)

        # Coach: LangGraph graph (--graph) OR the legacy heuristic cascade.
        if getattr(args, "graph", False) and _coach is not None:
            cr = _coach.run(metrics=m, env=env, history=history)
            nxt, nxt_reason = cr["next_env"], cr["reason"]
            # Print the per-node trace so the auto loop is fully observable.
            for line in cr.get("log", []):
                print(f"  {line}")
        else:
            nxt, nxt_reason = propose_next(cfg, env, m, args.llm, algo=algo)
        history.append({
            "iter": i, "metrics": m, "score": sc, "kept": kept,
            "reason": reason, "env": dict(env), "_next_reason": nxt_reason,
            "_next_env": dict(nxt),  # the config to apply next — needed to --resume cleanly
        })
        env = nxt  # apply the proposed tweak for the next iteration
        write_log(cfg, history)  # update the log every iter (resumable, observable)
        _refresh_db(cfg)  # keep the SQLite read-view in sync (events + this run)

    if not history:
        print("[autonomous] no iteration completed — nothing to report.")
        return

    best = max(history, key=lambda h: h["score"])
    print(f"\n[autonomous] DONE ({len(history)}/{args.iterations} iters ok). "
          f"Best iter {best['iter']} score {best['score']:.2f}.")

    # Final comprehensive report in 30-runs/ — always written if any iteration ran.
    report_path = write_final_report(cfg, history, doom_map, use_llm=args.llm)
    print(f"[autonomous] See:\n"
          f"  Live log    → {cfg.vault_path}/00-index/Autonomy Log.md\n"
          f"  Full report → {report_path}")


if __name__ == "__main__":
    main()
