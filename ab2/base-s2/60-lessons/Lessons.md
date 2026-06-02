---
type: lessons
created: 2026-06-02T06:20:52+00:00
events: 805
runs: 1
tags:
  - lessons
  - doom-rl
---

# Lessons learned (across runs)

From **805** episode events over **1** run(s): 805 deaths (100%), 0 successes.

## 1. Agents frequently die at low health levels

Low-HP deaths (health < 30) account for 92% of all deaths, indicating that agents often fail due to insufficient health.

_Evidence: 92% of deaths occur when the agent's health is below 30._

## 2. Agents rarely succeed and frequently die in short episodes

The mean episode length for successes (deaths) is significantly shorter than that for failures, suggesting a high failure rate.

_Evidence: Mean episode length for deaths: 101 steps vs Mean episode length for successes: 0 steps._

## 3. Agents often die near the beginning of episodes

Given the low-HP death frequency and short success episode lengths, agents may be dying early in their attempts.

_Evidence: Low-HP deaths account for 92% of all deaths._
