"""Progressive curriculum engine (V2 Phase 2).

The V1 mistake: training directly on full maps (hard). V2 starts on ViZDoom's
purpose-built scenarios (tiny, clear objectives) and advances to the full game:

  Stage 0 — MYWH:     my_way_home scenario — find the exit, no enemies, tiny map.
                       The easiest possible exit-finding task (exit-rate > 0 fast).
  Stage 1 — CORRIDOR: deadly_corridor scenario — navigate a corridor with enemies.
                       Trains survival before we add map complexity.
  Stage 2 — NAVIGATE: freedoom2 MAP01, combat zeroed — focus on map exploration.
  Stage 3 — FULL:     freedoom2 MAP01, all rewards — the complete task.

Stages 0–1 are SCENARIO mode (DoomEnv, built-in ViZDoom maps) and train their own
brain family. Stages 2–3 are CAMPAIGN mode (CampaignDoomEnv) and train a new brain.
The scenario stages prove exit-finding is learnable; the campaign stages apply it to
the real maps. Both algo choices (ppo / dqn) are supported at every stage.

    python -m rl.progressive_curriculum --stages mywh,corridor,navigate,full
    python -m rl.progressive_curriculum --algo dqn --steps-per-stage 200000
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Stage definitions ─────────────────────────────────────────────────────────
# Each entry: (mode, reward_profile_overrides)
# mode "scenario" → CAMPAIGN=0, DOOM_SCENARIO=<name> (uses DoomEnv)
# mode "campaign" → CAMPAIGN=1, MAPS=<doom_map>      (uses CampaignDoomEnv)

def _mywh_wad() -> str:
    """Absolute path to the my_way_home PWAD bundled with ViZDoom."""
    import vizdoom as vzd
    import os
    return os.path.join(vzd.scenarios_path, "my_way_home.wad")


STAGE_DEFS = {
    # ── my_way_home: campaign mode with the full action space ─────────────────────
    # Runs via CampaignDoomEnv (IWAD=freedoom2.wad + PWAD=my_way_home.wad, MAP01).
    # Same 15-19 actions as the full campaign → brain weights transfer directly to
    # MAP01/MAP02 runs with no compatibility issue (same action count = same network).
    # No enemies: agent focuses entirely on navigation / finding the exit.
    "mywh": {
        "_mode": "campaign",              # CampaignDoomEnv, NOT the scenario env
        "_scenario_wad_fn": "_mywh_wad", # resolved at runtime (avoids import at parse)
        "EPISODE_TIMEOUT": "2100",
        "EXIT_REWARD": "500.0",
        "COVERAGE_REWARD": "1.5",
        "FRONTIER_REWARD": "0.05",
        "USE_RND": "1", "RND_SCALE": "0.5",
        "DEATH_PENALTY": "1.0",
        # No combat rewards (no enemies in my_way_home)
        "HIT_REWARD": "0", "MISS_PENALTY": "0", "KILL_REWARD": "0",
        "DAMAGE_TAKEN_PENALTY": "0",
        # Assists off: this stage trains the real solo nav skill
        "AUTO_AIM": "0", "AUTO_DOOR_NAV": "0",
        "ENT_COEF": "0.05",
    },
    "corridor": {
        "_mode": "scenario",
        "_scenario": "deadly_corridor",   # corridor + enemies → trains survival
        "EPISODE_TIMEOUT": "2100",
        "EXIT_REWARD": "500.0",
        "HIT_REWARD": "2.0", "MISS_PENALTY": "0.05", "KILL_REWARD": "5.0",
        "DEATH_PENALTY": "15.0",
        "DAMAGE_TAKEN_PENALTY": "0.2",
        "ENT_COEF": "0.03",
        "USE_RND": "1", "RND_SCALE": "0.3",
    },
    # ── Campaign mode (freedoom2 full maps) ───────────────────────────────────────
    "navigate": {
        "_mode": "campaign",
        # Goal: find and reach the exit. Combat zeroed so the agent focuses on movement.
        "COVERAGE_REWARD": "2.0",
        "FRONTIER_REWARD": "0.1",
        "EXIT_REWARD": "1000.0",
        "EXIT_PROX_SCALE": "0.5",
        "USE_RND": "1", "RND_SCALE": "0.5",
        "HIT_REWARD": "0", "MISS_PENALTY": "0", "KILL_REWARD": "0",
        "DEATH_PENALTY": "1.0",
        "DAMAGE_TAKEN_PENALTY": "0.0",
        "EPISODE_TIMEOUT": "3500",
        "ENT_COEF": "0.05",
    },
    "survive": {
        "_mode": "campaign",
        # Goal: navigate without dying. Combat now costs.
        "COVERAGE_REWARD": "1.0",
        "FRONTIER_REWARD": "0.05",
        "EXIT_REWARD": "500.0",
        "EXIT_PROX_SCALE": "0.3",
        "USE_RND": "1", "RND_SCALE": "0.3",
        "HIT_REWARD": "2.0", "MISS_PENALTY": "0.05", "KILL_REWARD": "5.0",
        "DEATH_PENALTY": "15.0",
        "DAMAGE_TAKEN_PENALTY": "0.3",
        "EPISODE_TIMEOUT": "2800",
        "ENT_COEF": "0.03",
    },
    "full": {
        "_mode": "campaign",
        # Stage 3: restore normal defaults — memory and docs enabled.
    },
}

# ── Core ───────────────────────────────────────────────────────────────────────

# Campaign-only obs channels: the simple scenario env (doom/env.py DoomEnv) does NOT support
# game-vars (Dict obs), spatial memory, depth, automap or strafe — they're CampaignDoomEnv
# features. Forcing them off for every scenario stage avoids a MultiInputPolicy/obs-shape crash.
_SCENARIO_FORCE_OFF = {
    "GAME_VARS": "0", "SPATIAL_MEMORY": "0", "DEPTH_PERCEPTION": "0",
    "AUTOMAP": "0", "STRAFE": "0",
}

# Env vars that change the brain FAMILY (obs shape / action count / policy class).
# A campaign-mode stage profile must NEVER override these: the whole point of the
# campaign-mode mywh stage is that its weights transfer to the full-map stages, and
# transfer silently breaks the moment any of these differs between stages.
_FAMILY_KEYS = ("STRAFE", "SPATIAL_MEMORY", "DEPTH_PERCEPTION", "AUTOMAP",
                "GAME_VARS", "FRAME_STACK", "USE_LSTM", "SEMANTIC_CHANNEL")


def check_family_parity(stage: str, profile: dict) -> list[str]:
    """Family-affecting keys a campaign-mode stage profile illegally overrides.
    Empty list = weights transfer cleanly between this stage and the others."""
    if profile.get("_mode", "campaign") != "campaign":
        return []  # scenario stages have their own brain family by design
    return [k for k in _FAMILY_KEYS if k in profile]


def _run_stage(stage: str, profile: dict, doom_map: str,
               steps: int, algo: str, fresh: bool) -> None:
    mode     = profile.get("_mode", "campaign")
    scenario = profile.get("_scenario", "")
    # Transfer-parity preflight: a campaign stage overriding a family key would train
    # a DIFFERENT brain and the cross-stage transfer would silently never happen.
    bad = check_family_parity(stage, profile)
    if bad:
        raise SystemExit(
            f"[curriculum] stage '{stage}' overrides brain-family keys {bad} — "
            f"its weights could not transfer to the other campaign stages. "
            f"Set these via .env (globally) instead, so every stage agrees.")
    # Resolve lazy scenario-WAD path (avoids vizdoom import at module load time).
    scenario_wad = ""
    if "_scenario_wad_fn" in profile:
        fn_name = profile["_scenario_wad_fn"]
        import rl.progressive_curriculum as _self
        scenario_wad = getattr(_self, fn_name)()
    # Strip internal keys before passing to the subprocess env.
    env_overrides = {k: v for k, v in profile.items() if not k.startswith("_")}

    if mode == "scenario":
        # ViZDoom built-in scenario: use DoomEnv (CAMPAIGN=0). Campaign-only channels off.
        base_env = {**os.environ, "CAMPAIGN": "0",
                    "DOOM_SCENARIO": scenario,
                    "DOCS_ENABLED": "0", "MEMORY_ENABLED": "0",
                    **_SCENARIO_FORCE_OFF,
                    **env_overrides}
        if algo == "dqn":
            cmd = [sys.executable, "-m", "rl.train_dqn",
                   "--timesteps", str(steps), "--n-envs", "1"]
        else:
            cmd = [sys.executable, "-m", "rl.train",
                   "--timesteps", str(steps)]
    else:
        # Campaign mode: freedoom2 IWAD + optional scenario PWAD overlay.
        base_env = {**os.environ, "CAMPAIGN": "1", "MAPS": doom_map,
                    "DOCS_ENABLED": "0",
                    "MEMORY_ENABLED": "1" if stage == "full" else "0",
                    **env_overrides}
        if scenario_wad:
            # PWAD overlay: the stage map lives in the PWAD, not in freedoom2.wad.
            # CampaignDoomEnv loads IWAD as base and PWAD via set_doom_scenario_path.
            base_env["SCENARIO_WAD"] = scenario_wad
        if algo == "dqn":
            cmd = [sys.executable, "-m", "rl.train_dqn",
                   "--map", doom_map, "--timesteps", str(steps), "--n-envs", "1"]
        else:
            cmd = [sys.executable, "-m", "rl.train",
                   "--maps", doom_map, "--timesteps", str(steps)]

    if fresh:
        cmd.append("--fresh")

    tag = (f"scenario:{scenario}" if mode == "scenario"
           else f"map:{doom_map}" + (f"+pwad" if scenario_wad else ""))
    print(f"\n{'═'*64}")
    print(f"  STAGE: {stage.upper()}  |  algo: {algo}  |  steps: {steps:,}  |  {tag}")
    key = {k: v for k, v in env_overrides.items()
           if any(x in k for x in ("EXIT_REWARD", "DEATH_PENALTY", "COVERAGE_REWARD"))}
    if key:
        print(f"  reward profile: {key}")
    print(f"{'═'*64}\n")
    subprocess.run(cmd, cwd=ROOT, env=base_env, check=False)


def _eval_stage(profile: dict, doom_map: str, episodes: int = 20,
                algo: str = "ppo") -> dict:
    mode     = profile.get("_mode", "campaign")
    scenario = profile.get("_scenario", "")
    env_overrides = {k: v for k, v in profile.items() if not k.startswith("_")}

    if mode == "scenario":
        env = {**os.environ, "CAMPAIGN": "0", "DOOM_SCENARIO": scenario,
               "DOCS_ENABLED": "0", "MEMORY_ENABLED": "0",
               **_SCENARIO_FORCE_OFF, **env_overrides}
    else:
        env = {**os.environ, "CAMPAIGN": "1", "MAPS": doom_map,
               "DOCS_ENABLED": "0", "MEMORY_ENABLED": "0", **env_overrides}

    cmd = [sys.executable, "-m", "rl.eval",
           "--episodes", str(episodes), "--json", "--algo", algo]
    # QR-DQN ignores temperature (value-based); PPO uses it for the honest tempered score.
    if algo != "dqn":
        cmd += ["--temperature", "0.5"]
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("METRICS_JSON"):
            return json.loads(line.split("METRICS_JSON", 1)[1])
    return {}


def run(doom_map: str = "MAP01", steps_per_stage: int = 150_000,
        algo: str = "ppo", stages: list | None = None,
        eval_episodes: int = 20, fresh: bool = False) -> dict:
    """Run the full progressive curriculum and return per-stage metrics.

    fresh defaults to FALSE: rl.train auto-resumes the newest compatible brain (or
    starts from zero when none exists). The old `fresh=(i == 0)` default would have
    WIPED the long-trained campaign brain the moment a campaign-mode stage (mywh)
    ran first — weights are the only asset that compounds; never discard them by
    default. Pass --fresh only to deliberately restart a brain family."""
    stages = stages or ["mywh", "corridor", "navigate", "full"]
    out_dir = os.path.join(ROOT, "reports",
                           f"curriculum-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for i, stage in enumerate(stages):
        profile = STAGE_DEFS.get(stage, {})
        _run_stage(stage, profile, doom_map, steps_per_stage, algo,
                   fresh=(fresh and i == 0))

        print(f"\n── eval: {stage} ({profile.get('_mode','campaign')}) ──")
        m = _eval_stage(profile, doom_map, eval_episodes, algo=algo)
        results[stage] = m
        if m:
            print(f"  explored={m.get('explored_fraction',0):.0%}  "
                  f"exit={m.get('exit_rate',0):.0%}  "
                  f"→exit={m.get('exit_progress',0):.0%}  "
                  f"death={m.get('death_rate',0):.0%}  "
                  f"kills={m.get('kills_per_episode',0):.1f}")

        with open(os.path.join(out_dir, f"stage{i+1}_{stage}.json"), "w") as f:
            json.dump({"stage": stage, "metrics": m}, f, indent=2)

    _write_summary(out_dir, results, doom_map, algo)
    print(f"\n✅  curriculum done → {out_dir}/summary.md")
    return results


def _write_summary(out_dir, results, doom_map, algo) -> None:
    pct = {"exit_rate", "exit_progress", "explored_fraction", "death_rate"}
    keys = ["exit_rate", "exit_progress", "explored_fraction",
            "death_rate", "kills_per_episode"]
    lines = [f"# 📈 Curriculum — {doom_map} ({algo})", "",
             "| stage | " + " | ".join(keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for stage, m in results.items():
        cells = " | ".join(
            f"{m.get(k,0)*100:.0f}%" if k in pct else f"{m.get(k,0):.2f}"
            for k in keys)
        lines.append(f"| **{stage}** | {cells} |")
    lines += ["",
              "> Scenario stages (mywh/corridor) and campaign stages (navigate/survive/full) have",
              "> DIFFERENT action spaces (5 / 7 / 15 actions), so each trains its own brain — the",
              "> scenario stages PROVE a skill is learnable in isolation; weight transfer happens",
              "> only WITHIN the campaign stages (all 15-action, same brain family).",
              f"> Reproduce: `doom-cli curriculum2 --algo {algo}`"]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Progressive curriculum (V2 Phase 2).")
    p.add_argument("--map", default="MAP01")
    p.add_argument("--steps-per-stage", type=int, default=150_000)
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    p.add_argument("--stages", default="mywh,corridor,navigate,full",
                   help="Comma-separated stages: mywh,corridor,navigate,survive,full. "
                        "mywh/corridor use ViZDoom built-in scenarios; navigate/survive/full "
                        "use freedoom2 campaign mode.")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--fresh", action="store_true",
                   help="Restart the FIRST stage's brain from zero. Default is resume — "
                        "campaign-mode stages share the campaign brain family, and a "
                        "default fresh would wipe a long-trained brain.")
    args = p.parse_args()
    run(doom_map=args.map, steps_per_stage=args.steps_per_stage,
        algo=args.algo, stages=args.stages.split(","),
        eval_episodes=args.eval_episodes, fresh=args.fresh)


if __name__ == "__main__":
    main()
