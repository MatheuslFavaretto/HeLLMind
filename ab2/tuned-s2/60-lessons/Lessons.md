---
type: lessons
created: 2026-06-02T06:17:08+00:00
events: 834
runs: 1
tags:
  - lessons
  - doom-rl
---

# Lessons learned (across runs)

From **834** episode events over **1** run(s): 834 deaths (100%), 0 successes.

## 1. Agents frequently die in low health states

The majority, 91%, of deaths occurred when the agent's health was below 30 HP.

_Evidence: Low-HP deaths (health < 30): 91% of deaths_

## 2. Agents often die in short episodes

Episodes frequently end with death rather than success, as evidenced by the mean episode length just before death being significantly shorter than those that succeed.

_Evidence: Mean episode length — deaths 98 vs successes 0 steps_

## 3. Agents are often near empty ammo when dying

The agent frequently dies with very low ammo, which may indicate a critical failure mode related to weapon management.

_Evidence: Mean ammo just before death: 10.2_
