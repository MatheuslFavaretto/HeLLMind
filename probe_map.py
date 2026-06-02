"""Probe a single campaign map for combat density: boot, spam attack+turn, count
hits/kills over a short window. Run one map per process so a hanging map can be
killed by an external watchdog without taking the others down."""
import sys
from doom.campaign import CampaignDoomEnv, default_wad

doom_map = sys.argv[1]
env = CampaignDoomEnv(wad_path=default_wad(), doom_map=doom_map, episode_timeout=350)
obs, _ = env.reset()
atk = env._attack_idx
n = env.action_space.n
hits = kills = shots = damage_taken = steps = 0
# pattern: turn a bit, shoot a lot (so we discover enemies in view)
seq = [2, 4, 3, 4, 0, 4, 4, 1, 4]  # turn_left, attack, turn_right, attack...
for t in range(250):
    a = seq[t % len(seq)]
    if a >= n:
        a = atk or 0
    obs, r, done, trunc, info = env.step(a)
    d = info["doom"]["deltas"]
    hits += d.get("hitcount", 0)
    kills += d.get("killcount", 0)
    damage_taken += d.get("damage_taken", 0)
    if a == atk:
        shots += 1
    steps += 1
    if done:
        obs, _ = env.reset()
env.close()
print(f"MAP={doom_map} steps={steps} shots={shots} hits={int(hits)} "
      f"kills={int(kills)} dmg_taken={int(damage_taken)} OK")
