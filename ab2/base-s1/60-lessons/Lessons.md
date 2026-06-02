---
type: lessons
created: 2026-06-02T06:13:45+00:00
events: 820
runs: 1
tags:
  - lessons
  - doom-rl
---

# Lessons learned (across runs)

From **820** episode events over **1** run(s): 820 deaths (100%), 0 successes.

## 1. Agents frequently die at low health levels

In 92% of the deaths, agents had a mean health just before death below 30 HP.

_Evidence: Low-HP deaths (health < 30): 92% of deaths_

## 2. Agents often die near the end of episodes

The mean episode length for deaths is significantly longer than for successes, indicating frequent death occurrences towards the end of runs.

_Evidence: Mean episode length — deaths 99 vs successes 0 steps_

## 3. Agents frequently die in corridors

Given the lack of timeouts and only a few successes, agents are likely dying due to poor corridor navigation or combat strategies.

_Evidence: No timeouts reported; only 0 successes out of 820 runs_

## Related concepts

- [[Concept - Action Distribution Stability]]
- [[Concept - Cell Exploration Efficiency]]
