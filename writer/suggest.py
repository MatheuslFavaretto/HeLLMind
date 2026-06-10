"""Reward Suggestions (Phase 6): the LLM proposes reward-weight tweaks; a human approves.

Offline/post-training. Looks at the observed behavior (event aggregates + the latest
snapshot's accuracy) and proposes modest changes to the tunable reward weights, writing
them to `00-index/Reward Suggestions.md` with the current vs. suggested values and how
to apply them. NOTHING is changed automatically — you apply a suggestion by editing the
matching variable in `.env` and running again.

    python -m writer.suggest                 # suggest for the .env vault
    python -m writer.suggest --model qwen2.5:7b
"""
import argparse
import os
from datetime import datetime, timezone
from typing import Optional

from config import Config
from writer.memory_store import MemoryStore
from writer.reflect import aggregate_events
from writer.snapshot_log import SnapshotLog, log_path_for

# knob -> the matching .env variable name
ENV_VAR = {
    "hit_reward": "HIT_REWARD",
    "miss_penalty": "MISS_PENALTY",
    "damage_taken_penalty": "DAMAGE_TAKEN_PENALTY",
    "death_penalty": "DEATH_PENALTY",
}


def suggest(cfg: Config, model: Optional[str] = None) -> Optional[str]:
    """Write a reward-suggestions note from the observed behavior. Returns the path."""
    stats = aggregate_events(MemoryStore.read_events(cfg.memory_dir))
    if stats["total"] < cfg.min_events_for_lessons:
        print(f"[suggest] only {stats['total']} event(s) — not enough to advise yet.")
        return None
    # Pull the latest snapshot's accuracy (events don't carry it).
    snaps = SnapshotLog.read_all(log_path_for(cfg.pending_dir, cfg.run_name))
    if snaps:
        stats["shooting_accuracy"] = float(snaps[-1].get("shooting_accuracy", 0.0))

    weights = cfg.reward_weights()
    from writer.llm_client import LLMWriter

    llm = LLMWriter(model=model or cfg.llm_model, host=cfg.ollama_host,
                    num_ctx=cfg.llm_num_ctx, num_predict=cfg.llm_num_predict,
                    keep_alive=cfg.llm_keep_alive,
                    timeout=getattr(cfg, "llm_timeout", 120.0))
    try:
        res = llm.generate_reward_suggestions(stats, weights)
        tweaks = [t for t in res.tweaks if t.knob in ENV_VAR]
        summary = res.summary
    except Exception as e:
        print(f"[suggest] LLM failed ({e}); skipping.")
        return None
    if not tweaks:
        print("[suggest] no actionable tweaks proposed.")
        return None

    rows = ["| Knob (.env) | Current | Suggested | Why |", "|---|---|---|---|"]
    for t in tweaks:
        rows.append(
            f"| `{ENV_VAR[t.knob]}` | {weights.get(t.knob)} | **{t.suggested}** | {t.reason} |"
        )
    fm = "\n".join([
        "---", "type: reward-suggestions",
        f"created: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "tags:", "  - suggestions", "  - doom-rl", "---", ""])
    body = (
        "# Reward suggestions (human-approved)\n\n"
        f"{summary}\n\n"
        + "\n".join(rows)
        + "\n\n> ⚠️ Not applied automatically. To accept, set the variable(s) above in "
        "`.env` and run again (use `--fresh` to retrain from scratch, or keep the brain "
        "to fine-tune).\n"
    )
    out_dir = os.path.join(cfg.vault_path, cfg.dir_index)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Reward Suggestions.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(fm + "\n" + body)
    print(f"[suggest] wrote {len(tweaks)} suggestion(s) -> "
          f"{os.path.join(cfg.dir_index, 'Reward Suggestions.md')}")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Propose reward-weight tweaks (human approves).")
    p.add_argument("--model", default=None, help="Ollama model (override).")
    args = p.parse_args()
    suggest(Config(), model=args.model)


if __name__ == "__main__":
    main()
