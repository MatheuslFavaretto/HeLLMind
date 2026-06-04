# 📊 HeLLMind ablation benchmark

_map MAP01 · 50,000 steps · 2 seeds · eval 20 eps (tempered)_

Does each layer add value? Mean ± std across seeds.

| config | exit_rate | explored_fraction | kills_per_episode | death_rate | combat_engagement | mean_base_reward |
|---|---|---|---|---|---|---|
| **baseline** | 0±0% | 4±0% | 1.07±0.12 | 68±2% | 29±0% | -24.06±0.84 |
| **rnd** | 0±0% | 4±0% | 1.35±0.05 | 62±7% | 22±1% | -22.68±1.28 |
| **memory** | 0±0% | 4±0% | 1.73±0.22 | 25±20% | 43±4% | -25.23±2.27 |
| **full** | 0±0% | 4±0% | 1.43±0.07 | 45±5% | 31±5% | -23.51±0.04 |

> Reproduce: `doom-cli benchmark`. Raw numbers in `benchmark.json/.csv`.
