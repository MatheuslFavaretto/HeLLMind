---
type: reward-suggestions
created: 2026-06-02T06:13:52+00:00
tags:
  - suggestions
  - doom-rl
---

# Reward suggestions (human-approved)

Tweaks to reward shaping weights based on observed behavior.

| Knob (.env) | Current | Suggested | Why |
|---|---|---|---|
| `HIT_REWARD` | 0.0 | **0.2** | Low shooting accuracy suggests the agent might be hesitant to shoot, warranting a slight increase in hit_reward to encourage more aggressive actions. |
| `DAMAGE_TAKEN_PENALTY` | 0.0 | **0.15** | High rate of low-HP deaths indicates that damage taken is not sufficiently penalized, suggesting an increase in damage_taken_penalty to discourage taking excessive damage. |
| `DEATH_PENALTY` | 0.0 | **2.0** | Given the high death rate and significant portion of low-HP deaths, increasing death_penalty will reinforce the agent's avoidance of death, which is crucial for survival in Doom. |

> ⚠️ Not applied automatically. To accept, set the variable(s) above in `.env` and run again (use `--fresh` to retrain from scratch, or keep the brain to fine-tune).
