---
type: reward-suggestions
created: 2026-06-02T06:24:34+00:00
tags:
  - suggestions
  - doom-rl
---

# Reward suggestions (human-approved)

Tweaks to reward shaping weights based on observed behavior.

| Knob (.env) | Current | Suggested | Why |
|---|---|---|---|
| `HIT_REWARD` | 1.0 | **1.0** | Current hit_reward is low, but increasing it too much could lead to aggressive playstyle which might not be optimal for the current observed accuracy. |
| `MISS_PENALTY` | 0.1 | **0.2** | Accuracy is very low (8%), suggesting a need to penalize misses more heavily to encourage better aim and reduce errors. |
| `DAMAGE_TAKEN_PENALTY` | 0.05 | **0.15** | The observed death rate is high, with many deaths due to low health. Increasing the penalty for damage taken can help incentivize players to maintain higher health throughout the game. |

> ⚠️ Not applied automatically. To accept, set the variable(s) above in `.env` and run again (use `--fresh` to retrain from scratch, or keep the brain to fine-tune).
