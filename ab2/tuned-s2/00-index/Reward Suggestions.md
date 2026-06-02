---
type: reward-suggestions
created: 2026-06-02T06:17:15+00:00
tags:
  - suggestions
  - doom-rl
---

# Reward suggestions (human-approved)

Tweaks to reward shaping weights based on observed behavior.

| Knob (.env) | Current | Suggested | Why |
|---|---|---|---|
| `HIT_REWARD` | 1.0 | **1.2** | To encourage more accurate shooting, which is currently very low at only 8%. |
| `DAMAGE_TAKEN_PENALTY` | 0.05 | **0.1** | Given the high rate of low-HP deaths (91%) and mean health just before death of 13.1, increasing penalty for damage taken could help reduce these instances. |
| `DEATH_PENALTY` | 5.0 | **6.0** | To further emphasize the importance of avoiding deaths, which are occurring at a high rate (100%). |

> ⚠️ Not applied automatically. To accept, set the variable(s) above in `.env` and run again (use `--fresh` to retrain from scratch, or keep the brain to fine-tune).
