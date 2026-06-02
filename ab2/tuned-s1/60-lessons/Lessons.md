---
type: lessons
created: 2026-06-02T06:10:34+00:00
events: 858
runs: 1
tags:
  - lessons
  - doom-rl
---

# Lessons learned (across runs)

From **858** episode events over **1** run(s): 858 deaths (100%), 0 successes.

## 1. Agents frequently die in low health

The majority, 91%, of deaths occur when the agent's health is below 30 HP.

_Evidence: Low-HP deaths (health < 30): 91% of deaths_

## 2. Agents often die in short episodes

Episodes end with a death much faster than they succeed, indicating the agent's performance is precarious.

_Evidence: Mean episode length — deaths 95 vs successes 0 steps_

## 3. Agent tends to die near the beginning of its life

The mean health just before death (13.7) suggests agents often fail early in their lives.

_Evidence: Mean health just before death: 13.7_

## Related concepts

- [[Concept - Action Distribution Stability]]
