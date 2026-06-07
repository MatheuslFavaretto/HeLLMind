"""LangGraph-based coach (V2 Phase 4).

Replaces the ad-hoc propose() / propose_next() cascade in autonomous.py with an explicit
stateful graph where every decision is a named node — observable, debuggable, extensible.

Graph:
  OBSERVE → DIAGNOSE → HYPOTHESIZE → PROPOSE → VALIDATE → ADOPT / REVERT → DONE

Each node receives the full CoachState and returns a partial update. The graph is compiled
once and called every auto-loop iteration instead of propose_next().

Usage:
    from rl.coach_graph import CoachGraph
    coach = CoachGraph(cfg, use_llm=True)
    result = coach.run(metrics=m, env=current_env, history=history)
    next_env = result["next_env"]
    reason   = result["reason"]
"""
import operator
from typing import Any, Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END


# ── State ─────────────────────────────────────────────────────────────────────

class CoachState(TypedDict):
    # Inputs
    metrics:  dict[str, Any]          # eval metrics for this iteration
    env:      dict[str, Any]          # current reward config (env-var dict)
    history:  list[dict]              # prior iterations
    use_llm:  bool
    memory_dir: str
    algo:     str                     # "ppo" or "dqn" — selects the un-freeze knob

    # Populated by nodes
    diagnosis:   str                  # what's wrong (e.g. "passive_combat")
    hypothesis:  str                  # the proposed change and why
    next_env:    dict[str, Any]       # the tweaked config to apply next
    reason:      str                  # human-readable explanation
    score:       float                # composite score for this iteration
    kept:        bool                 # was the change kept?

    # Accumulate log messages across nodes
    log: Annotated[list[str], operator.add]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score(m: dict) -> float:
    """Composite score — same formula as autonomous.score()."""
    exit_r    = float(m.get("exit_rate", 0.0))
    exit_prog = float(m.get("exit_progress", 0.0))
    explored  = float(m.get("explored_fraction", 0.0))
    kills     = min(float(m.get("kills_per_episode", 0.0)), 5.0) / 5.0
    accuracy  = float(m.get("shooting_accuracy", 0.0))
    return 4.0 * exit_r + 1.5 * exit_prog + 3.0 * explored + 1.0 * accuracy + 0.5 * kills


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_observe(state: CoachState) -> dict:
    """Compute the composite score for this iteration's metrics."""
    m = state["metrics"]
    sc = _score(m)
    return {
        "score": sc,
        "log": [f"[observe] score={sc:.3f}  explored={m.get('explored_fraction',0):.0%}  "
                f"exit={m.get('exit_rate',0):.0%}  →exit={m.get('exit_progress',0):.0%}  "
                f"death={m.get('death_rate',0):.0%}  "
                f"combat_engagement={m.get('combat_engagement',0):.0%}"],
    }


def node_diagnose(state: CoachState) -> dict:
    """Classify the failure mode. One clear diagnosis drives the proposal."""
    m = state["metrics"]
    timeout_rate = float(m.get("timeout_rate", 0.0))
    explored     = float(m.get("explored_fraction", 0.0))
    death_rate   = float(m.get("death_rate", 0.0))
    kills        = float(m.get("kills_per_episode", 0.0))
    engagement   = m.get("combat_engagement")  # None if USE_LABELS is off

    if timeout_rate > 0.80 and explored < 0.15:
        diag = "episode_too_short"
    elif explored < 0.10:
        diag = "stuck_at_spawn"
    elif death_rate > 0.5:
        diag = "dying_too_much"
    elif (engagement is not None and float(m.get("combat_fraction", 0)) > 0.05
          and float(engagement) < 0.3):
        diag = "passive_in_combat"
    elif kills < 0.5 and timeout_rate < 0.6:
        diag = "passive_overall"
    elif explored < 0.5:
        diag = "low_exploration"
    elif float(m.get("shooting_accuracy", 0)) < 0.10:
        diag = "poor_aim"
    elif float(m.get("exit_rate", 0)) == 0.0:
        diag = "never_exits"
    else:
        diag = "healthy"

    return {"diagnosis": diag,
            "log": [f"[diagnose] diagnosis={diag}"]}


def node_hypothesize(state: CoachState) -> dict:
    """Turn the diagnosis into a falsifiable hypothesis string."""
    diag = state["diagnosis"]
    # The un-freeze knob differs by algorithm: PPO uses entropy, QR-DQN uses ε-greedy.
    unfreeze = "DQN_EPS_FINAL" if state.get("algo") == "dqn" else "ENT_COEF"
    hypotheses = {
        "episode_too_short":  "Raise EPISODE_TIMEOUT — agent times out before exploring",
        "stuck_at_spawn":     "Raise COVERAGE_REWARD + FRONTIER_REWARD + RND_SCALE — push agent away from spawn",
        "dying_too_much":     "Raise DAMAGE_TAKEN_PENALTY + DEATH_PENALTY — teach it to survive",
        "passive_in_combat":  f"Raise {unfreeze} + ENGAGEMENT_REWARD — unfreeze policy when enemies are visible",
        "passive_overall":    f"Raise {unfreeze} + ENGAGEMENT_REWARD — policy argmax has collapsed",
        "low_exploration":    "Raise COVERAGE_REWARD — exploration is stuck at mid-level",
        "poor_aim":           "Raise MISS_PENALTY + HIT_REWARD — tighten aim incentive",
        "never_exits":        "Raise EXIT_REWARD — agent never reaches the level end",
        "healthy":            "Anneal COVERAGE_REWARD — consolidate the policy",
    }
    hyp = hypotheses.get(diag, "Unknown failure mode — hold config")
    return {"hypothesis": hyp,
            "log": [f"[hypothesize] {hyp}"]}


def node_propose(state: CoachState) -> dict:
    """Apply the heuristic delta + memory + optional LLM on top."""
    from rl.autonomous import propose, BOUNDS

    m, env, diag = state["metrics"], state["env"], state["diagnosis"]

    # Re-use the tested heuristic (propose returns a safe, clamped delta). Pass the algo so
    # the un-freeze knob matches the engine (PPO→ENT_COEF, DQN→DQN_EPS_FINAL).
    new, heuristic_reason = propose(env, m, algo=state.get("algo", "ppo"))

    # Layer memory on top (death patterns → knob suggestions that avoid regressed changes)
    reason = heuristic_reason
    try:
        from writer.memory_policy import recall_proposal
        mem_env, mem_reason = recall_proposal(state["memory_dir"], new)
        if mem_env is not None:
            new, reason = mem_env, f"{reason}; memory: {mem_reason}"
    except Exception:
        pass

    # Layer SEMANTIC memory: 'have I seen a situation like this before, and what worked?' Fills in
    # proven params for knobs the heuristic didn't already target (the heuristic's targeted fix
    # wins; semantic recall supplies validated priors for the rest).
    try:
        from rl.autonomous import semantic_recall
        from rl.tuning_registry import validate
        recalled, sem_note = semantic_recall(state["memory_dir"], m)
        if recalled:
            touched = {k for k in new if str(new.get(k)) != str(env.get(k))}  # heuristic's changes
            fill = {k: v for k, v in recalled.items() if k not in touched}
            if fill:
                new = validate(fill, base_env=new)   # add proven priors, clamped; keep heuristic's
                reason = f"{reason}; {sem_note}"
    except Exception:
        pass

    # Layer LLM on top if enabled. Prefer the OPEN proposer (full parameter catalog — the LLM
    # can change ANY knob it wants, validated against the registry); fall back to the combat-only
    # proposer if the open one has nothing/Ollama is down.
    if state.get("use_llm"):
        try:
            from rl.autonomous import llm_propose, llm_propose_open
            from config import Config
            cfg = Config(); cfg.memory_dir = state["memory_dir"]
            res = llm_propose_open(cfg, new, m) or llm_propose(cfg, new, m)
            if res:
                new, llm_r = res
                reason = f"{reason}; {llm_r}"
        except Exception:
            pass

    return {"next_env": new, "reason": reason,
            "log": [f"[propose] {reason[:120]}"]}


def node_validate(state: CoachState) -> dict:
    """Decide keep/revert based on score vs best-so-far (with 5% tolerance)."""
    history = state["history"]
    best_so_far = max((h.get("score", 0.0) for h in history), default=-1e9)
    sc = state["score"]
    kept = (not history) or (sc >= best_so_far - 0.05)
    return {"kept": kept,
            "log": [f"[validate] score={sc:.3f} best={best_so_far:.3f} → "
                    f"{'KEEP ✅' if kept else 'REVERT ↩'}"]}


def node_adopt(state: CoachState) -> dict:
    """Kept: record in the rollback log."""
    try:
        from writer.rollback import RollbackLog, diff_envs
        env = state["env"]; nxt = state["next_env"]
        before, change, after = diff_envs(env, nxt)
        RollbackLog(state["memory_dir"]).record(
            len(state["history"]), before, change, after,
            {"score": state["score"]}, kept=True)
        # Remember this (situation → change → good outcome) so a similar future state recalls it.
        from rl.autonomous import semantic_record
        semantic_record(state["memory_dir"], state["metrics"], change,
                        state["score"], kept=True, reason=state.get("reason", ""))
    except Exception:
        pass
    return {"log": ["[adopt] config kept and logged (+ semantic memory)"]}


def node_revert(state: CoachState) -> dict:
    """Reverted: roll next_env back to the last good config and log."""
    history = state["history"]
    last_good = history[-1]["env"] if history else state["env"]
    try:
        from writer.rollback import RollbackLog, diff_envs
        before, change, after = diff_envs(state["env"], state["next_env"])
        RollbackLog(state["memory_dir"]).record(
            len(history), before, change, after,
            {"score": state["score"]}, kept=False)
        # Remember the regression too — so a similar future state won't be told it "worked".
        from rl.autonomous import semantic_record
        semantic_record(state["memory_dir"], state["metrics"], change,
                        state["score"], kept=False, reason=state.get("reason", ""))
    except Exception:
        pass
    return {"next_env": last_good,
            "log": ["[revert] regression — rolled back to last good config"]}


# ── Edge functions ─────────────────────────────────────────────────────────────

def route_validate(state: CoachState) -> str:
    return "adopt" if state["kept"] else "revert"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    g = StateGraph(CoachState)
    for name, fn in [("observe", node_observe), ("diagnose", node_diagnose),
                     ("hypothesize", node_hypothesize), ("propose", node_propose),
                     ("validate", node_validate), ("adopt", node_adopt),
                     ("revert", node_revert)]:
        g.add_node(name, fn)

    g.set_entry_point("observe")
    g.add_edge("observe",     "diagnose")
    g.add_edge("diagnose",    "hypothesize")
    g.add_edge("hypothesize", "propose")
    g.add_edge("propose",     "validate")
    g.add_conditional_edges("validate", route_validate, {"adopt": "adopt", "revert": "revert"})
    g.add_edge("adopt",  END)
    g.add_edge("revert", END)
    return g.compile()


# Compiled once at import time
_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ── Public API ─────────────────────────────────────────────────────────────────

class CoachGraph:
    """Drop-in replacement for the ad-hoc propose_next() call in the auto loop."""

    def __init__(self, cfg, use_llm: bool = False, algo: str = "ppo"):
        self.cfg = cfg
        self.use_llm = use_llm
        self.algo = algo

    def run(self, metrics: dict, env: dict, history: list) -> dict:
        """Run the full coach graph for one iteration.

        Returns a dict with:
          next_env  — the config to apply next iteration
          reason    — human-readable explanation of what changed and why
          score     — composite score for this iteration
          kept      — whether the change was kept or reverted
          log       — list of per-node log messages (for debugging / Obsidian)
          diagnosis — the classified failure mode
        """
        initial: CoachState = {
            "metrics": metrics, "env": env, "history": history,
            "use_llm": self.use_llm, "memory_dir": self.cfg.memory_dir,
            "algo": self.algo,
            "diagnosis": "", "hypothesis": "", "next_env": dict(env),
            "reason": "", "score": 0.0, "kept": True, "log": [],
        }
        result = _get_graph().invoke(initial)
        return {
            "next_env":  result["next_env"],
            "reason":    result["reason"],
            "score":     result["score"],
            "kept":      result["kept"],
            "log":       result["log"],
            "diagnosis": result["diagnosis"],
            "hypothesis": result["hypothesis"],
        }

    def draw(self, path: str = "coach_graph.png") -> None:
        """Render the graph to a PNG (needs graphviz + pillow)."""
        try:
            img = _get_graph().get_graph().draw_mermaid_png()
            with open(path, "wb") as f:
                f.write(img)
            print(f"[coach_graph] saved → {path}")
        except Exception as exc:
            print(f"[coach_graph] draw failed: {exc}")
