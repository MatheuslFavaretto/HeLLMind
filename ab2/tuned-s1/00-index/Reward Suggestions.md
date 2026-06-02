---
type: reward-suggestions
created: 2026-06-02T06:10:41+00:00
tags:
  - suggestions
  - doom-rl
---

# Reward suggestions (human-approved)

Tweaks to reward shaping weights based on observed behavior.

| Knob (.env) | Current | Suggested | Why |
|---|---|---|---|
| `HIT_REWARD` | 1.0 | **1.2** | The low shooting accuracy suggests the agent may be hesitant or inaccurate in its shots, warranting a slight increase in hit_reward to encourage more aggressive and accurate shooting. |
| `DAMAGE_TAKEN_PENALTY` | 0.05 | **0.15** | Given that many deaths occur with low health (91% of low-HP deaths), increasing the penalty for taking damage could incentivize the agent to maintain higher health levels and avoid getting into dangerous situations. |
| `DEATH_PENALTY` | 5.0 | **6.0** | The high death rate (100%) indicates that even when the agent does not take significant damage, it still frequently dies. Raising the death penalty could encourage the agent to avoid risky situations or seek out safer paths. |

> ⚠️ Not applied automatically. To accept, set the variable(s) above in `.env` and run again (use `--fresh` to retrain from scratch, or keep the brain to fine-tune).
