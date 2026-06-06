"""Headless ViZDoom env smoke tests (real engine, no window).

These boot the actual ViZDoom engine but render nothing (window_visible=False, tiny
resolution) and run only a handful of steps, so they stay fast. Skipped automatically
where ViZDoom or its scenarios/WAD aren't available (e.g. a minimal CI image), so the
pure-logic suite still runs everywhere.
"""
import pytest

vzd = pytest.importorskip("vizdoom")

from doom.campaign import CampaignDoomEnv, default_wad  # noqa: E402
from doom.env import DoomEnv  # noqa: E402


def _scenario_available(name: str) -> bool:
    import os
    return os.path.exists(os.path.join(vzd.scenarios_path, f"{name}.cfg"))


# --------------------------- scenario env ---------------------------
@pytest.mark.skipif(not _scenario_available("defend_the_center"),
                    reason="defend_the_center scenario not bundled")
def test_scenario_env_contract():
    env = DoomEnv(scenario="defend_the_center")
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        for _ in range(8):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert obs.shape == env.observation_space.shape
            assert isinstance(float(reward), float)
            doom = info.get("doom")
            assert doom is not None and "deltas" in doom and "levels" in doom
            if terminated or truncated:
                # Regression: scenario terminals are 3-way (death / timeout / exit). Reaching
                # the goal early (e.g. my_way_home) must label "exit", NOT "timeout" — that
                # mislabel hid the project's first exit-rate > 0.
                assert doom["terminal"] in ("death", "timeout", "exit")
                env.reset()
    finally:
        env.close()


def test_classify_terminal_three_way():
    """Regression: reaching the goal early (not dead, before the timeout) = 'exit', NOT
    'timeout'. That mislabel hid the project's first exit-rate > 0 on my_way_home."""
    from doom.env import classify_terminal
    assert classify_terminal(is_dead=True, episode_time=10, episode_timeout=2100) == "death"
    assert classify_terminal(is_dead=False, episode_time=10, episode_timeout=2100) == "exit"
    assert classify_terminal(is_dead=False, episode_time=2100, episode_timeout=2100) == "timeout"
    assert classify_terminal(is_dead=False, episode_time=2500, episode_timeout=2100) == "timeout"
    # No timeout configured (0) → ending alive is always a goal-reach, never a timeout.
    assert classify_terminal(is_dead=False, episode_time=999, episode_timeout=0) == "exit"


# --------------------------- campaign env ---------------------------
@pytest.mark.skipif(not __import__("os").path.exists(default_wad()),
                    reason="freedoom2.wad not bundled")
def test_campaign_env_contract_and_spatial_channel():
    env = CampaignDoomEnv(wad_path=default_wad(), doom_map="MAP01",
                          episode_timeout=300, spatial_memory=True)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape[-1] == 2  # spatial memory adds a 2nd channel
        assert info["map"] == "MAP01"
        saw_walls = False
        for _ in range(10):
            obs, reward, done, trunc, info = env.step(env.action_space.sample())
            assert obs.shape == env.observation_space.shape
            doom = info["doom"]
            saw_walls = saw_walls or ("walls" in doom)
            if done:
                assert doom["terminal"] in ("death", "exit", "timeout")
                break
        assert saw_walls  # geometry is emitted once for the real minimap
    finally:
        env.close()


@pytest.mark.skipif(not __import__("os").path.exists(default_wad()),
                    reason="freedoom2.wad not bundled")
def test_auto_use_pulses_not_holds():
    # Regression (found on watch: agent stood at a door and it never opened). Doom's USE is
    # EDGE-triggered — holding it every frame opens a door only ONCE. auto-USE must PULSE
    # (alternate on/off across steps) so each contact makes a fresh key-down edge.
    env = CampaignDoomEnv(wad_path=default_wad(), doom_map="MAP01",
                          rewards={"auto_use": 1.0})
    try:
        env.reset(seed=0)
        assert env._use_idx is not None
        # Action 0 (FWD) never presses USE, so auto-USE drives the pulse. Step a few times and
        # collect the forced USE state — it must ALTERNATE, not stay held at 1.
        states = []
        for _ in range(6):
            env.step(0)
            states.append(bool(env._use_held))
        assert any(states) and not all(states), f"USE didn't pulse: {states}"
        # Specifically: consecutive steps must differ (a fresh edge each press).
        assert any(states[i] != states[i + 1] for i in range(len(states) - 1))
    finally:
        env.close()


@pytest.mark.skipif(not __import__("os").path.exists(default_wad()),
                    reason="freedoom2.wad not bundled")
def test_campaign_weapon_variety_seeds_spawn_weapon():
    # With the variety reward on, the spawn weapon must be pre-seeded so it doesn't
    # pay out every episode just for holding the starting pistol.
    env = CampaignDoomEnv(wad_path=default_wad(), doom_map="MAP01",
                          rewards={"weapon_variety_reward": 0.5})
    try:
        env.reset(seed=0)
        spawn = int(env._last_vars.get("selected_weapon", 0.0))
        assert spawn in env._weapons_seen  # spawn weapon already counted
    finally:
        env.close()
