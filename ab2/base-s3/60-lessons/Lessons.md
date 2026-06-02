---
type: lessons
created: 2026-06-02T06:27:42+00:00
events: 816
runs: 1
tags:
  - lessons
  - doom-rl
---

# Lessons learned (across runs)

From **816** episode events over **1** run(s): 816 deaths (100%), 0 successes.

## 1. Agents frequently die in low health states

The majority (91%) of deaths occur when the agent's health is below 30 HP.

_Evidence: Low-HP deaths: 91%_

## 2. Agent tends to die near episode end

Most deaths happen close to the end of an episode, with a mean length of 100 steps before death.

_Evidence: Mean episode length — deaths: 100_

## 3. Agent frequently runs out of ammo

The agent's ammo levels are often depleted just before dying, with a mean of 7.7 ammo at the time of death.

_Evidence: Mean ammo just before death: 7.7_
