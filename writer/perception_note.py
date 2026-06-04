"""Writes a static 'How the agent perceives the world' note to the vault.

One-time call, idempotent. Explains pixels + game vars + objects_info to anyone
reading the knowledge graph — and to the LLM when it generates lessons.

    python -m writer.perception_note
"""
import os


CONTENT = """\
---
type: concept
title: How the agent perceives Doom
tags: [perception, architecture, doom-rl]
---

# How the agent perceives Doom

The agent has **no semantic labels**. It learns entirely from pixels + reward signals.

## What it sees

### 1. Raw pixels (84×84, 4 stacked frames)
The only visual input. The agent cannot read the HUD, cannot identify objects by name,
cannot distinguish "enemy" from "wall" explicitly. It learns that certain pixel patterns
correlate with positive or negative rewards.

### 2. Game variables (numeric, every step)
ViZDoom exposes a set of numeric state variables at each step:

| Variable | What it measures |
|----------|-----------------|
| `HEALTH` | Current health (0–100) |
| `AMMO2` | Current ammo count |
| `KILLCOUNT` | Total kills this episode |
| `HITCOUNT` | Cumulative hits on enemies |
| `DAMAGECOUNT` | Cumulative damage taken |
| `SELECTED_WEAPON` | Weapon slot (1–7) |
| `POSITION_X/Y/Z` | Map coordinates |

These are used to compute deltas (e.g. `kill_delta`, `damage_delta`) which drive the
reward signal. The agent receives the delta, not the raw value.

### 3. Objects info (map-wide, every step)
`set_objects_info_enabled(True)` exposes **all actors** in the current map:
- Class name (e.g. `"DoomImp"`, `"Medikit"`, `"Shotgun"`)
- Position (x, y, z)
- Velocity (from which approach direction is inferred)
- Health

This is how the **bestiary** works: the code in `doom/entities.py` classifies actor
names into `MONSTERS`, `PROJECTILE_CASTER`, `HITSCAN` sets, and the `_track_enemies()`
method accumulates per-monster statistics (encounters, kills, distance, who killed the agent).

## What the agent does NOT know explicitly

| Question | Reality |
|----------|---------|
| Where is the exit? | Discovered by accident — large reward when reached |
| What is this item? | Learned empirically (walk toward it → health rises = health pack) |
| Where are the doors? | No explicit door detection — USE button pressed through trial/error |
| Is this enemy dangerous? | Learned from `DAMAGECOUNT` delta when near it |
| Which weapon is best? | Learned from `HITCOUNT` delta per weapon slot vs kill rate |

## How the reward teaches perception

```
Agent walks toward a green sprite
→ HEALTH increases by 25
→ reward += 0  (health bonus not explicitly rewarded, but death is penalised)
→ agent learns: "go toward green things when low health"
```

```
Agent faces a moving sprite
→ fires, HITCOUNT+1, KILLCOUNT+1
→ reward += HIT_REWARD + KILL_REWARD
→ agent learns: "these pixel patterns = shootable"
```

```
Agent reaches the level exit
→ episode ends, not dead, before timeout
→ reward += EXIT_REWARD (1000)
→ agent learns: "the path to that area = very good"
```

## What this means for behaviour

- **Circling**: the agent found that moving (any movement) gives `MOVE_REWARD`. Circling
  maximises distance traveled → maximises move_reward → a local optimum.
- **Passivity**: when `MOVE_REWARD=0` and the starting room's cells are all visited,
  standing still has zero negative reward. The argmax policy freezes.
- **Exit-rate = 0%**: without ever accidentally reaching the exit, the agent has no
  positive signal toward it. Exploration is the prerequisite.
- **Doors not opened**: `USE` button is in the action space but the reward for opening
  a door (new area → coverage reward) is only discovered if the agent happens to press
  USE near a door — a very sparse event.

## Links

- [[Concept - Exploration vs Exploitation]]
- [[Concept - Reward Shaping]]
- [[Bestiary]] (world model from objects_info)
- [[Knowledge Graph]]
"""


def write(cfg) -> str:
    out_dir = os.path.join(cfg.vault_path, cfg.dir_concepts)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Concept - Agent Perception.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(CONTENT)
    return path


def main() -> None:
    from config import Config
    cfg = Config()
    path = write(cfg)
    print(f"[perception] wrote {path}")


if __name__ == "__main__":
    main()
