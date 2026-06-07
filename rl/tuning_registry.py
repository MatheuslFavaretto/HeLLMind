"""The full catalog of TUNABLE parameters — the single source the LLM/auto-loop reads so it
KNOWS every knob it can turn, what each does, and the safe range. Lets the coach propose changes
to ANY of them (validated against bounds), not just a hardcoded combat subset.

Each entry: env var, (lo, hi) bounds, type, group, and a one-line description of its EFFECT.
"""
from typing import Dict, List, Tuple

# group, env, lo, hi, type, description (what turning it UP does)
_P = [
    # --- combat reward ---
    ("combat", "KILL_REWARD", 2.0, 20.0, "float", "reward per kill — pays for finishing enemies"),
    ("combat", "HIT_REWARD", 1.0, 10.0, "float", "reward per shot that LANDS — pays for aim"),
    ("combat", "MISS_PENALTY", 0.0, 0.3, "float", "penalty per shot that misses — discourages spray"),
    ("combat", "ENGAGEMENT_REWARD", 0.0, 0.4, "float", "reward for keeping an enemy CENTRED — teaches aiming"),
    # --- survival ---
    ("survival", "DEATH_PENALTY", 2.0, 30.0, "float", "penalty for dying — pushes it to not die"),
    ("survival", "DAMAGE_TAKEN_PENALTY", 0.0, 0.5, "float", "penalty per HP lost — pushes it to dodge/retreat"),
    ("survival", "LIVING_REWARD", -0.1, 0.1, "float", "per-step reward just for being alive (small)"),
    # --- exploration reward ---
    ("explore", "COVERAGE_REWARD", 0.0, 3.0, "float", "reward for stepping on a NEW cell — drives exploration"),
    ("explore", "FRONTIER_REWARD", 0.0, 0.3, "float", "reward for NET outward progress — anti-circling"),
    ("explore", "RND_SCALE", 0.0, 1.0, "float", "intrinsic curiosity strength — explores novel places"),
    ("explore", "DISCOVERY_REWARD", 0.0, 1.0, "float", "reward first sighting of a new object (key/item)"),
    ("explore", "MOVE_REWARD", 0.0, 0.1, "float", "reward per unit moved (anti-idle; can farm circling)"),
    ("explore", "GOEXPLORE_GOAL_PROB", 0.0, 0.8, "float", "prob an episode is sent to a far frontier cell"),
    ("explore", "GOEXPLORE_GOAL_SCALE", 0.0, 0.1, "float", "reward scale for reaching the frontier goal"),
    # --- objective ---
    ("objective", "EXIT_REWARD", 0.0, 1500.0, "float", "big prize for reaching the level EXIT"),
    ("objective", "EXIT_PROX_SCALE", 0.0, 2.0, "float", "dense reward for getting closer to a known exit"),
    ("objective", "COVERAGE_CELL", 32.0, 192.0, "float", "grid cell size for coverage/exploration"),
    # --- combat/explore arbitration ---
    ("arbitration", "COMBAT_EXPLORE_FACTOR", 0.05, 1.0, "float", "how much off-mode reward survives (0.1=suppress)"),
    # --- policy exploration ---
    ("policy", "ENT_COEF", 0.005, 0.1, "float", "PPO entropy bonus — un-freezes a collapsed policy"),
    ("policy", "DQN_EPS_FINAL", 0.02, 0.3, "float", "QR-DQN final epsilon — exploration floor"),
    # --- training hyperparams ---
    ("train", "EPISODE_TIMEOUT", 1050.0, 8400.0, "int", "max ticks per episode — time to find things"),
    ("train", "BATCH_SIZE", 64.0, 1024.0, "int", "PPO minibatch size"),
    ("train", "N_STEPS", 256.0, 4096.0, "int", "PPO rollout length before each update"),
    ("train", "ENT_COEF", 0.005, 0.1, "float", "(see policy) entropy bonus"),
]

# De-duplicate (ENT_COEF listed twice for grouping) keeping the first.
PARAMS: List[Dict] = []
_seen = set()
for grp, env, lo, hi, typ, desc in _P:
    if env in _seen:
        continue
    _seen.add(env)
    PARAMS.append({"group": grp, "env": env, "lo": lo, "hi": hi, "type": typ, "desc": desc})

_BY_ENV = {p["env"]: p for p in PARAMS}


def all_bounds() -> Dict[str, Tuple[float, float]]:
    """env -> (lo, hi) for every tunable param (the guardrail covers ALL of them)."""
    return {p["env"]: (p["lo"], p["hi"]) for p in PARAMS}


def clamp(env_key: str, value: float):
    """Clamp a value to its registered bounds (ints rounded). Unknown key → value unchanged."""
    p = _BY_ENV.get(env_key)
    if not p:
        return value
    v = max(p["lo"], min(p["hi"], float(value)))
    return int(round(v)) if p["type"] == "int" else round(v, 4)


def validate(proposal: Dict[str, float], base_env: Dict[str, str] = None) -> Dict[str, str]:
    """Keep only KNOWN params, clamp each to bounds, return env-string dict. Drops unknowns so a
    hallucinated/typo'd knob can't slip through — the LLM can propose anything; only valid,
    in-range params are applied."""
    out: Dict[str, str] = dict(base_env or {})
    for k, v in (proposal or {}).items():
        if k in _BY_ENV:
            try:
                out[k] = str(clamp(k, float(v)))
            except (TypeError, ValueError):
                continue
    return out


def describe_for_llm(current_env: Dict[str, str] = None) -> str:
    """A catalog the LLM can read: every knob, its CURRENT value, range, and what it does —
    grouped, so the model knows the whole action space it may tune."""
    current_env = current_env or {}
    lines = ["TUNABLE PARAMETERS (you may propose a new value for ANY of these, within range):"]
    last_group = None
    for p in PARAMS:
        if p["group"] != last_group:
            lines.append(f"\n[{p['group']}]")
            last_group = p["group"]
        cur = current_env.get(p["env"], "?")
        rng = (f"{int(p['lo'])}..{int(p['hi'])}" if p["type"] == "int"
               else f"{p['lo']}..{p['hi']}")
        lines.append(f"  {p['env']} = {cur}  (range {rng}) — {p['desc']}")
    return "\n".join(lines)
