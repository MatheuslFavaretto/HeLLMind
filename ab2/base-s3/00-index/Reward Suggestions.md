---
type: reward-suggestions
created: 2026-06-02T06:27:49+00:00
tags:
  - suggestions
  - doom-rl
---

# Reward suggestions (human-approved)

Tweaking reward weights to improve accuracy and reduce low-HP deaths.

| Knob (.env) | Current | Suggested | Why |
|---|---|---|---|
| `HIT_REWARD` | 0.0 | **0.2** | To incentivize better shooting accuracy, given the observed 7% hit rate. |
| `DAMAGE_TAKEN_PENALTY` | 0.0 | **1.5** | To reduce deaths due to low health, with a moderate increase from current value of 0.0. |
| `DEATH_PENALTY` | 0.0 | **2.0** | To further discourage death occurrences, given the high death rate and significant portion of low-HP deaths. |

> ⚠️ Not applied automatically. To accept, set the variable(s) above in `.env` and run again (use `--fresh` to retrain from scratch, or keep the brain to fine-tune).
