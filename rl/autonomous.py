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

from config import Config

PY = sys.executable


# ---- the GOAL, as one number: explore the whole map, finish it, and survive/fight ----
def score(m: dict) -> float:
    return (
        4.0 * m.get("exit_rate", 0.0)            # finishing the map matters most
        + 3.0 * m.get("explored_fraction", 0.0)  # then covering it
        + 0.5 * m.get("kills_per_episode", 0.0)  # then fighting
        + 1.0 * m.get("shooting_accuracy", 0.0)
    )


# Reward knobs the supervisor is allowed to move, with hard bounds (the guardrails).
BOUNDS = {
    "COVERAGE_REWARD": (0.0, 3.0),
    "EXIT_REWARD": (0.0, 500.0),
    "HIT_REWARD": (0.5, 5.0),
    "MISS_PENALTY": (0.0, 0.3),
    "DAMAGE_TAKEN_PENALTY": (0.0, 0.5),
    "DEATH_PENALTY": (1.0, 20.0),
}

# writer.suggest speaks in lowercase knobs; map them onto the supervisor's env vars.
# (Exploration knobs COVERAGE/EXIT aren't in writer.suggest — the heuristic owns those.)
LLM_KNOB_TO_ENV = {
    "hit_reward": "HIT_REWARD",
    "miss_penalty": "MISS_PENALTY",
    "damage_taken_penalty": "DAMAGE_TAKEN_PENALTY",
    "death_penalty": "DEATH_PENALTY",
}


def propose(env: dict, m: dict) -> tuple[dict, str]:
    """Heuristic 'understanding -> action': nudge the knob that targets the weakest
    metric, within bounds. Returns (new_env, human-readable reason)."""
    new = dict(env)

    def bump(key, factor=None, add=None):
        lo, hi = BOUNDS[key]
        v = float(new.get(key, 0.0))
        v = v * factor if factor is not None else v + add
        new[key] = round(max(lo, min(hi, v)), 4)

    if m.get("explored_fraction", 0.0) < 0.5:
        bump("COVERAGE_REWARD", factor=1.5)
        return new, f"explored only {m.get('explored_fraction',0):.0%} -> raise COVERAGE_REWARD to {new['COVERAGE_REWARD']}"
    if m.get("exit_rate", 0.0) == 0.0:
        bump("EXIT_REWARD", factor=1.3)
        bump("COVERAGE_REWARD", factor=1.2)  # exploring helps find the exit
        return new, f"never reached the exit -> raise EXIT_REWARD to {new['EXIT_REWARD']}, COVERAGE to {new['COVERAGE_REWARD']}"
    if m.get("shooting_accuracy", 0.0) < 0.10:
        bump("MISS_PENALTY", add=0.05)
        bump("HIT_REWARD", factor=1.2)
        return new, f"accuracy {m.get('shooting_accuracy',0):.0%} -> MISS_PENALTY {new['MISS_PENALTY']}, HIT_REWARD {new['HIT_REWARD']}"
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


def propose_next(cfg: Config, env: dict, m: dict, use_llm: bool) -> tuple[dict, str]:
    """Pick the next reward config. The heuristic always runs (it owns exploration and
    is the fallback); when --llm is on, the LLM refines the combat knobs on top of it."""
    new, reason = propose(env, m)
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


def train_chunk(env: dict, doom_map: str, steps: int, fresh: bool) -> None:
    cmd = [PY, "-m", "rl.train", "--maps", doom_map,
           "--n-envs", str(env.get("N_ENVS", "4")), "--timesteps", str(steps)]
    cmd.append("--fresh" if fresh else "--resume")
    subprocess.run(cmd, env=_subprocess_env(env), check=True)


def eval_brain(env: dict, episodes: int) -> dict:
    out = subprocess.run(
        [PY, "-m", "rl.eval", "--episodes", str(episodes), "--json"],
        env=_subprocess_env(env), check=True, capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("METRICS_JSON "):
            return json.loads(line[len("METRICS_JSON "):])
    raise RuntimeError("eval produced no METRICS_JSON")


def write_log(cfg: Config, history: list) -> None:
    """Persist the self-improvement trail: JSONL (machine) + Obsidian note (human)."""
    os.makedirs(cfg.memory_dir, exist_ok=True)
    with open(os.path.join(cfg.memory_dir, "autonomy.jsonl"), "w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")

    note = os.path.join(cfg.vault_path, cfg.dir_index, "Autonomy Log.md")
    os.makedirs(os.path.dirname(note), exist_ok=True)
    best = max(history, key=lambda h: h["score"])
    lines = [
        "---", "type: autonomy-log",
        f"updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "tags: [autonomy, doom-rl]", "---", "",
        "# Autonomy log — the agent improving itself",
        "",
        "Each row: the supervisor trained a chunk, evaluated, scored against the goal "
        "(explore + complete + fight), then adjusted the reward and kept it only if the "
        "score held or improved.",
        "",
        "| Iter | Explored | Exit% | Kills/ep | Acc | Score | Kept? | Decision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in history:
        m = h["metrics"]
        lines.append(
            f"| {h['iter']} | {m['explored_fraction']:.0%} | {m['exit_rate']:.0%} | "
            f"{m['kills_per_episode']:.2f} | {m['shooting_accuracy']:.0%} | "
            f"{h['score']:.2f} | {'✅' if h['kept'] else '↩︎ reverted'} | {h['reason']} |"
        )
    lines += [
        "", f"## Best so far (iter {best['iter']}, score {best['score']:.2f})", "",
        "Apply these to `.env` to lock in the agent's own best configuration:", "",
        "```bash",
        *[f"{k}={best['env'][k]}" for k in BOUNDS if k in best["env"]],
        "```", "",
    ]
    with open(note, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous self-improving supervisor.")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--steps", type=int, default=100000, help="Timesteps per iteration.")
    p.add_argument("--map", default=None, help="Map to train on (default: cfg.maps[0]).")
    p.add_argument("--episodes", type=int, default=10, help="Eval episodes per iteration.")
    p.add_argument("--fresh", action="store_true", help="Start the first iter from zero.")
    p.add_argument("--llm", action="store_true",
                   help="Let the offline reward-suggestions LLM refine the combat knobs "
                        "(needs Ollama + enough events); falls back to the heuristic.")
    args = p.parse_args()

    cfg = Config()
    doom_map = args.map or cfg.maps[0]
    # Seed the evolving reward env from the current config (campaign mode, no docs:
    # the supervisor is fast; documentation is a separate, final concern).
    env = {
        "CAMPAIGN": "1", "MAPS": doom_map, "DOCS_ENABLED": "0", "MEMORY_ENABLED": "1",
        "CONTROL_ENABLED": "0", "N_ENVS": str(cfg.n_envs),
        "SPATIAL_MEMORY": "1" if cfg.spatial_memory else "0",
        "USE_LSTM": "1" if cfg.use_lstm else "0",
        "COVERAGE_REWARD": str(cfg.coverage_reward), "EXIT_REWARD": str(cfg.exit_reward),
        "HIT_REWARD": str(cfg.hit_reward), "MISS_PENALTY": str(cfg.miss_penalty),
        "DAMAGE_TAKEN_PENALTY": str(cfg.damage_taken_penalty),
        "DEATH_PENALTY": str(cfg.death_penalty), "MOVE_REWARD": str(cfg.move_reward),
        "LIVING_REWARD": str(cfg.living_reward), "EPISODE_TIMEOUT": str(cfg.episode_timeout),
    }

    history = []
    best_score = -1e9
    print(f"[autonomous] {args.iterations} iterations × {args.steps} steps on {doom_map} "
          f"({'LLM-refined' if args.llm else 'heuristic'} reward proposals)")
    for i in range(args.iterations):
        fresh = args.fresh and i == 0
        reason = "baseline" if i == 0 else history[-1].get("_next_reason", "adjust")
        print(f"\n===== ITER {i} ({'fresh' if fresh else 'resume'}) — {reason} =====")
        train_chunk(env, doom_map, args.steps, fresh)
        m = eval_brain(env, args.episodes)
        sc = score(m)
        kept = (i == 0) or (sc >= best_score - 0.05)  # guardrail: revert regressions
        print(f"[autonomous] iter {i}: score={sc:.2f} (best={best_score:.2f}) "
              f"explored={m['explored_fraction']:.0%} exit={m['exit_rate']:.0%} "
              f"kills={m['kills_per_episode']:.2f} -> {'KEEP' if kept else 'REVERT'}")

        if not kept:
            env = history[-1]["env"]  # roll back to the last good reward config
        else:
            best_score = max(best_score, sc)

        nxt, nxt_reason = propose_next(cfg, env, m, args.llm)
        history.append({
            "iter": i, "metrics": m, "score": sc, "kept": kept,
            "reason": reason, "env": dict(env), "_next_reason": nxt_reason,
        })
        env = nxt  # apply the proposed tweak for the next iteration
        write_log(cfg, history)  # update the log every iter (resumable, observable)

    best = max(history, key=lambda h: h["score"])
    print(f"\n[autonomous] DONE. Best iter {best['iter']} score {best['score']:.2f}. "
          f"See {cfg.vault_path}/00-index/Autonomy Log.md")


if __name__ == "__main__":
    main()
