"""Eureka-style reward evolution — the LLM as a reward DESIGNER, not just a knob-nudger.

Eureka (NVIDIA, 2023) had an LLM write reward functions, run them, read the results, and
rewrite — an evolutionary search over reward designs with reflection. This is the safe,
RL-Doom flavour of that idea: instead of executing arbitrary LLM-written code, candidates
are structured reward-weight VECTORS. Each generation:

    1. PROPOSE  — the LLM mutates the current best into N candidates, *reasoning* over the
                  history of (config → measured score) pairs (reflection). No Ollama? fall
                  back to bounded random mutation around the best.
    2. EVALUATE — train each candidate a short chunk, eval deterministically, score it
                  against the GOAL (reuses rl.autonomous.score).
    3. SELECT   — keep the best; it seeds the next generation.

The pure pieces (mutation, clamping, selection) are unit-tested; the train/eval orchestration
reuses the autonomy engine.

    python -m rl.eureka --generations 3 --pop 4 --steps 50000 --map MAP01
"""
import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

from config import Config

sys.stdout.reconfigure(line_buffering=True)

# Reward knobs Eureka may design, with hard bounds (the guardrails). Superset of the
# autonomy combat knobs — Eureka also owns the exploration levers.
EUREKA_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HIT_REWARD":           (0.5, 5.0),
    "MISS_PENALTY":         (0.0, 0.3),
    "DAMAGE_TAKEN_PENALTY": (0.0, 0.5),
    "DEATH_PENALTY":        (1.0, 20.0),
    "KILL_REWARD":          (1.0, 10.0),
    "MOVE_REWARD":          (0.0, 0.01),
    "LIVING_REWARD":        (-0.05, 0.0),
    "COVERAGE_REWARD":      (0.0, 4.0),
    "FRONTIER_REWARD":      (0.0, 0.2),
    "EXIT_REWARD":          (0.0, 2000.0),
}


def clamp_env(env: Dict[str, str]) -> Dict[str, str]:
    """Clamp every tunable knob in `env` to its bound (values kept as strings)."""
    out = dict(env)
    for k, (lo, hi) in EUREKA_BOUNDS.items():
        if k in out:
            try:
                v = float(out[k])
            except (TypeError, ValueError):
                continue
            out[k] = str(round(max(lo, min(hi, v)), 5))
    return out


def mutate_heuristic(base: Dict[str, str], rng: Optional[random.Random] = None,
                     strength: float = 0.3) -> Dict[str, str]:
    """No-LLM fallback: perturb each tunable knob by ±strength (multiplicative), clamped.
    A bounded random walk in reward space — the evolutionary 'mutation' operator."""
    rng = rng or random
    out = dict(base)
    for k, (lo, hi) in EUREKA_BOUNDS.items():
        cur = float(out.get(k, lo))
        span = hi - lo
        # Mix multiplicative jitter with a little absolute jitter so zeros can escape 0.
        delta = cur * rng.uniform(-strength, strength) + span * 0.05 * rng.uniform(-1, 1)
        out[k] = str(round(max(lo, min(hi, cur + delta)), 5))
    return out


def select_best(candidates: List[Dict]) -> Optional[Dict]:
    """Pick the highest-scoring evaluated candidate. Each item: {"env", "score", ...}."""
    scored = [c for c in candidates if c.get("score") is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: c["score"])


def _candidate_schema(n: int) -> dict:
    """JSON schema for the LLM: a list of N reward-weight candidates, each with reasoning."""
    knob_props = {k: {"type": "number"} for k in EUREKA_BOUNDS}
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        **knob_props,
                    },
                    "required": ["reasoning"],
                },
            }
        },
        "required": ["candidates"],
    }


def propose_candidates(cfg: Config, base_env: Dict[str, str], metrics: dict,
                       history: List[Dict], n: int,
                       rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    """Return N candidate reward configs. Uses the LLM (reflecting over history) when Ollama
    is reachable; otherwise bounded random mutation around the base."""
    rng = rng or random
    # Try the LLM first.
    try:
        from ollama import Client
        client = Client(host=cfg.ollama_host, timeout=300.0)
        system = (
            "You are a reinforcement-learning reward ENGINEER for a Doom agent. The goal, in "
            "priority order: reach the level EXIT, explore the whole map, then fight. You are "
            "given the current reward weights, the latest measured metrics, and a history of "
            "past (weights -> score) results. Propose diverse candidate weight sets that you "
            "predict will raise the score. Reason briefly. Stay within plausible ranges."
        )
        hist_lines = []
        for h in history[-6:]:
            hist_lines.append(f"  score={h.get('score'):.3f}  "
                              + " ".join(f"{k}={h['env'].get(k)}" for k in EUREKA_BOUNDS
                                         if k in h.get("env", {})))
        user = (
            "Current weights:\n" + "\n".join(f"  {k}={base_env.get(k)}" for k in EUREKA_BOUNDS) +
            f"\n\nLatest metrics: explored={metrics.get('explored_fraction',0):.0%} "
            f"exit_rate={metrics.get('exit_rate',0):.0%} "
            f"kills={metrics.get('kills_per_episode',0):.2f} "
            f"timeout_rate={metrics.get('timeout_rate',0):.0%}\n\n"
            f"History (most recent last):\n" + ("\n".join(hist_lines) or "  (none)") +
            f"\n\nPropose {n} candidate weight sets."
        )
        resp = client.chat(
            model=cfg.llm_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            format=_candidate_schema(n),
            options={"temperature": 0.7, "num_ctx": cfg.llm_num_ctx, "num_predict": 800},
            keep_alive=cfg.llm_keep_alive,
        )
        data = json.loads(resp.message.content)
        cands = []
        for c in data.get("candidates", [])[:n]:
            env = dict(base_env)
            for k in EUREKA_BOUNDS:
                if k in c and isinstance(c[k], (int, float)):
                    env[k] = str(c[k])
            cands.append(clamp_env(env))
        if cands:
            print(f"[eureka] LLM proposed {len(cands)} candidate(s)")
            return cands
    except Exception as e:
        print(f"[eureka] LLM unavailable ({type(e).__name__}) — using heuristic mutation")

    # Fallback: bounded random mutation around the base.
    return [clamp_env(mutate_heuristic(base_env, rng)) for _ in range(n)]


def evolve(cfg: Config, doom_map: str, generations: int, pop_size: int,
           steps: int, episodes: int) -> Optional[Dict]:
    """Run the evolutionary reward search. Reuses the autonomy train/eval/score engine."""
    from rl.autonomous import eval_brain, score, train_chunk

    base_env = {
        "CAMPAIGN": "1", "MAPS": doom_map, "DOCS_ENABLED": "0", "MEMORY_ENABLED": "1",
        "CONTROL_ENABLED": "0", "N_ENVS": str(cfg.n_envs),
        "HIT_REWARD": str(cfg.hit_reward), "MISS_PENALTY": str(cfg.miss_penalty),
        "DAMAGE_TAKEN_PENALTY": str(cfg.damage_taken_penalty),
        "DEATH_PENALTY": str(cfg.death_penalty), "KILL_REWARD": str(cfg.kill_reward),
        "MOVE_REWARD": str(cfg.move_reward), "LIVING_REWARD": str(cfg.living_reward),
        "COVERAGE_REWARD": str(cfg.coverage_reward), "FRONTIER_REWARD": str(cfg.frontier_reward),
        "EXIT_REWARD": str(cfg.exit_reward), "EPISODE_TIMEOUT": str(cfg.episode_timeout),
    }
    rng = random.Random(cfg.seed)
    history: List[Dict] = []
    best: Optional[Dict] = None
    metrics: dict = {}

    for gen in range(generations):
        print(f"\n===== EUREKA GEN {gen} =====")
        seed_env = best["env"] if best else base_env
        candidates = propose_candidates(cfg, seed_env, metrics, history, pop_size, rng)
        for ci, cand_env in enumerate(candidates):
            full = {**base_env, **cand_env}
            fresh = (gen == 0 and ci == 0 and best is None)
            print(f"[eureka] gen {gen} cand {ci}: training {steps} steps...")
            try:
                train_chunk(full, doom_map, steps, fresh)
                temp = cfg.eval_temperature if cfg.eval_temperature > 0 else None
                metrics = eval_brain(full, episodes, temperature=temp)
            except Exception as e:
                print(f"[eureka] candidate failed ({type(e).__name__}): {e}")
                continue
            sc = score(metrics)
            entry = {"gen": gen, "cand": ci, "env": dict(full), "score": sc,
                     "metrics": metrics}
            history.append(entry)
            print(f"[eureka] gen {gen} cand {ci}: score={sc:.3f} "
                  f"explored={metrics.get('explored_fraction',0):.0%} "
                  f"exit={metrics.get('exit_rate',0):.0%}")
            if best is None or sc > best["score"]:
                best = entry
                print(f"[eureka] new BEST score={sc:.3f}")
        _write_log(cfg, history, best, doom_map)

    if best:
        print(f"\n[eureka] DONE. Best score {best['score']:.3f} (gen {best['gen']}).")
    return best


def _write_log(cfg: Config, history: List[Dict], best: Optional[Dict], doom_map: str) -> None:
    os.makedirs(cfg.memory_dir, exist_ok=True)
    with open(os.path.join(cfg.memory_dir, "eureka.jsonl"), "w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")
    if not best:
        return
    note_dir = os.path.join(cfg.vault_path, cfg.dir_runs)
    os.makedirs(note_dir, exist_ok=True)
    lines = [
        "---", "type: eureka-evolution", f"map: {doom_map}",
        f"evaluations: {len(history)}", f"best_score: {round(best['score'], 3)}", "---",
        f"# Eureka reward evolution — {doom_map}", "",
        f"Evaluated **{len(history)}** reward designs. Best score "
        f"**{round(best['score'], 3)}** (gen {best['gen']}).", "",
        "## Best reward weights", "```",
    ]
    for k in EUREKA_BOUNDS:
        if k in best["env"]:
            lines.append(f"{k}={best['env'][k]}")
    lines += ["```", "", "## Score trajectory", ""]
    lines.append(" → ".join(f"{round(h['score'], 2)}" for h in history))
    with open(os.path.join(note_dir, f"Eureka — {doom_map}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Eureka-style reward evolution.")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--pop", type=int, default=4, help="Candidates per generation.")
    p.add_argument("--steps", type=int, default=50000, help="Train steps per candidate.")
    p.add_argument("--episodes", type=int, default=10, help="Eval episodes per candidate.")
    p.add_argument("--map", default=None)
    args = p.parse_args()

    cfg = Config()
    doom_map = args.map or cfg.maps[0]
    print(f"[eureka] {args.generations} generations × {args.pop} candidates × "
          f"{args.steps} steps on {doom_map}")
    best = evolve(cfg, doom_map, args.generations, args.pop, args.steps, args.episodes)
    if best:
        print(f"[eureka] See: {cfg.vault_path}/{cfg.dir_runs}/Eureka — {doom_map}.md")


if __name__ == "__main__":
    main()
