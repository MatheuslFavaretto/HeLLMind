"""The ViZDoom GameVariables we extract on every step.

Core idea: capture LOTS of signal beyond the reward. We split the variables into:
- MONOTONIC: counters that only go up within an episode (kills, damage, etc). For
  these we report the per-step DELTA — so we can sum across a window even when
  episodes reset in the middle.
- LEVELS: instantaneous values (health, ammo). We report the current value and sample
  it for mean/min over the window.
"""
import vizdoom as vzd

# The ORDER here defines the order of `state.game_variables`. Don't reorder blindly.
TRACKED_VARS = [
    vzd.GameVariable.KILLCOUNT,       # enemies killed
    vzd.GameVariable.HITCOUNT,        # shots that landed
    vzd.GameVariable.HITS_TAKEN,      # times the agent was hit
    vzd.GameVariable.DAMAGECOUNT,     # total damage dealt
    vzd.GameVariable.DAMAGE_TAKEN,    # total damage taken
    vzd.GameVariable.DEATHCOUNT,      # deaths
    vzd.GameVariable.ITEMCOUNT,       # items picked up
    vzd.GameVariable.HEALTH,          # current health (level)
    vzd.GameVariable.AMMO2,           # starting-weapon ammo (level)
    vzd.GameVariable.POSITION_X,      # map position (for path/coverage)
    vzd.GameVariable.POSITION_Y,      # map position (for path/coverage)
    vzd.GameVariable.SELECTED_WEAPON, # selected weapon (slot)
]

VAR_NAMES = [
    "killcount", "hitcount", "hits_taken", "damagecount", "damage_taken",
    "deathcount", "itemcount", "health", "ammo2",
    "position_x", "position_y", "selected_weapon",
]

# Cumulative counters -> we report a per-step delta
MONOTONIC = [
    "killcount", "hitcount", "hits_taken", "damagecount", "damage_taken",
    "deathcount", "itemcount",
]
# Instantaneous values -> we report the current level (mean/min over the window)
LEVELS = ["health", "ammo2", "position_x", "position_y", "selected_weapon"]

assert len(TRACKED_VARS) == len(VAR_NAMES)
