---
type: reward-suggestions
created: 2026-06-02T06:21:00+00:00
tags:
  - suggestions
  - doom-rl
---

# Reward suggestions (human-approved)

Tweaks to reward shaping weights based on observed behavior.

| Knob (.env) | Current | Suggested | Why |
|---|---|---|---|
| `HIT_REWARD` | 0.0 | **0.5** | Low shooting accuracy suggests the agent might be hesitant to shoot, warranting a slight increase in hit_reward to encourage more aggressive behavior. |
| `DAMAGE_TAKEN_PENALTY` | 0.0 | **1.0** | High rate of low-HP deaths indicates the current penalty for taking damage is insufficient. Increasing damage_taken_penalty will make it more costly to take damage, potentially reducing the frequency of low-HP situations. |
| `DEATH_PENALTY` | 0.0 | **2.0** | Given the high death rate and many deaths occurring at low health, increasing death_penalty will further discourage the agent from dying, especially when it is close to death. |

> ⚠️ Not applied automatically. To accept, set the variable(s) above in `.env` and run again (use `--fresh` to retrain from scratch, or keep the brain to fine-tune).
