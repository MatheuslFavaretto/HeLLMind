"""Entrypoint: trains PPO on Doom and (optionally) documents into Obsidian.

Usage:
    python -m rl.train                 # training + notes in the vault (Ollama)
    python -m rl.train --no-docs       # pure training, no LLM/notes (lighter)
    python -m rl.train --render        # opens the Doom window (1 env, slower)
    python -m rl.train --render --no-docs --timesteps 100000

Flags override the .env.
"""
import argparse
import glob
import os
from typing import Optional

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)

from config import Config
from doom.campaign import campaign_metadata, make_campaign_env
from doom.env import make_doom_env, probe_env_metadata
from instrumentation.stats_tracker import StatsTracker
from rl.callbacks import DoomDocumentationCallback
from rl.campaign_callbacks import MapCurriculumCallback
from rl.control import ControlCallback
from rl.memory_callback import MemoryRecorderCallback
from writer.memory_store import MemoryStore
from writer.snapshot_log import (
    SnapshotLog,
    log_path_for,
    meta_path_for,
    write_meta,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on Doom + Obsidian notes.")
    p.add_argument(
        "--no-docs",
        action="store_true",
        help="Don't call the LLM or write notes (pure training, lighter).",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Open the Doom window (forces 1 env, non-parallel, slower).",
    )
    p.add_argument("--model", type=str, default=None, help="Ollama model (override).")
    p.add_argument("--timesteps", type=int, default=None, help="Total timesteps.")
    p.add_argument("--n-envs", type=int, default=None, help="Number of parallel envs.")
    p.add_argument(
        "--campaign",
        action="store_true",
        help="Campaign mode: play full WAD maps, in order.",
    )
    p.add_argument(
        "--maps",
        type=str,
        default=None,
        help="Comma-separated map list (e.g. MAP01,MAP02 or E1M1,E1M2).",
    )
    p.add_argument("--wad", type=str, default=None, help="WAD path (campaign).")
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from a specific .zip. By default the vault's brain is ALREADY "
        "reused automatically; use this only to point at a file.",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore the saved brain and start from ZERO (overwrites the vault's).",
    )
    p.add_argument(
        "--no-assists",
        dest="no_assists",
        action="store_true",
        help="Disable ALL gameplay assists (auto-aim, auto-door-nav, auto-best-weapon, "
             "auto-use). Trains a SOLO brain that must learn everything itself. "
             "Required for a policy that works assists-OFF in eval.",
    )
    return p.parse_args()


def apply_args(cfg: Config, args: argparse.Namespace) -> Config:
    if args.no_docs:
        cfg.docs_enabled = False
    if args.render:
        cfg.render = True
    if args.model:
        cfg.llm_model = args.model
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.campaign:
        cfg.campaign = True
    if args.maps:
        cfg.maps = tuple(args.maps.split(","))
    if args.wad:
        cfg.wad_path = args.wad
    if getattr(args, "no_assists", False):
        cfg.auto_aim = False
        cfg.auto_best_weapon = False
        cfg.auto_use = False
        cfg.auto_door_nav = False
    # Render needs a single windowed env; you can't watch parallel subprocesses.
    if cfg.render and cfg.n_envs != 1:
        print("[render] forcing n_envs=1 to show the Doom window.")
        cfg.n_envs = 1
    return cfg


def _probe_ollama(cfg: Config) -> bool:
    """Check Ollama WITHOUT crashing training. Returns True if it's ready.

    The project works perfectly without Ollama: snapshots are collected anyway and,
    at the end, notes come out in factual mode (no LLM narrative).
    """
    try:
        from ollama import Client

        client = Client(host=cfg.ollama_host)
        models = [m.model for m in client.list().models]
    except Exception:
        print(
            f"[docs] Ollama unavailable at {cfg.ollama_host} — continuing anyway.\n"
            f"       Notes will come out FACTUAL (no narrative). For the narrative,\n"
            f"       start `ollama serve` and run later: python -m writer.process_run"
        )
        return False
    wanted = cfg.llm_model if ":" in cfg.llm_model else cfg.llm_model + ":latest"
    if not any((m or "") == wanted for m in models):
        print(
            f"[docs] Model '{cfg.llm_model}' not found in Ollama (pull it with "
            f"`ollama pull {cfg.llm_model}`). Notes will be FACTUAL."
        )
        return False
    print(f"[docs] Ollama OK at {cfg.ollama_host} | model: {cfg.llm_model}")
    return True


def _latest_checkpoint(cfg: Config, name_prefix: str) -> Optional[str]:
    """The most recent brain for this task in the vault — by MODIFICATION TIME across both
    `_final.zip` and `_<digits>_steps.zip`.

    Matches only this exact family. The step files are `{name_prefix}_<digits>_steps.zip`,
    so we glob the digit boundary — otherwise `ppo_campaign_a8` (a prefix of
    `ppo_campaign_a8_lstm`) would wrongly pick up a RecurrentPPO brain and crash the load.

    Picking by mtime (not "prefer _final") matters: if a run is killed before its final
    save, `_final.zip` is STALE and a newer `_<steps>.zip` exists — preferring _final would
    silently resume the OLD brain and throw away the newer progress.
    """
    candidates = glob.glob(os.path.join(cfg.checkpoint_dir, f"{name_prefix}_[0-9]*_steps.zip"))
    final = os.path.join(cfg.checkpoint_dir, f"{name_prefix}_final.zip")
    if os.path.exists(final):
        candidates.append(final)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _latest_vecnorm(cfg: Config, name_prefix: str) -> Optional[str]:
    """The most recent saved VecNormalize stats (`{name_prefix}_vecnormalize_*.pkl`)."""
    cands = glob.glob(os.path.join(cfg.checkpoint_dir, f"{name_prefix}_vecnormalize_*.pkl"))
    return max(cands, key=os.path.getmtime) if cands else None


def _resolve_resume(cfg: Config, args: argparse.Namespace, name_prefix: str) -> Optional[str]:
    """By DEFAULT reuse the vault's brain. --fresh starts from zero; --resume PATH
    uses a specific file."""
    if args.fresh:
        return None
    if args.resume and args.resume != "auto":
        path = args.resume if args.resume.endswith(".zip") else args.resume + ".zip"
        return path if os.path.exists(path) else None
    return _latest_checkpoint(cfg, name_prefix)  # default: continue where it stopped


def _best_device() -> str:
    """Pick the best available compute device: CUDA > MPS (Apple Silicon) > CPU.
    SB3's 'auto' doesn't select MPS on M-series Macs — we pick it explicitly so the M5
    GPU is actually used instead of leaving it idle."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _lr_setting(cfg: Config):
    """Learning-rate value or a linear-decay schedule with a FLOOR (floats are captured,
    not the cfg, so it stays picklable for SB3 save/load).

    The floor matters on resume: SB3's progress_remaining is GLOBAL with
    reset_num_timesteps=False — `1 - num/(num + chunk)` — so a chunk resumed on an
    18M-step brain starts at progress≈0.02 and the unfloored schedule trained at
    ~0.02%→0% of base LR. The brain was effectively FROZEN on every resumed chunk
    (measured: lr 5e-06→9e-08, approx_kl 1.6e-05, clip_fraction 0)."""
    base = float(cfg.learning_rate)
    if not getattr(cfg, "lr_schedule", False):
        return base
    floor = max(0.0, min(1.0, float(getattr(cfg, "lr_min_factor", 0.1))))
    return lambda progress_remaining: base * max(progress_remaining, floor)


def build_vec_env(cfg: Config, normalize: bool = False):
    rewards = cfg.reward_weights()
    # Closed loop: scale kill rewards by the learned per-monster threat (bestiary -> reward).
    if cfg.campaign and getattr(cfg, "bestiary_reward", False):
        try:
            from writer.bestiary import BestiaryStore, threat_multipliers

            mults = threat_multipliers(BestiaryStore(cfg.memory_dir).load())
            if mults:
                rewards["enemy_threat"] = mults
                print(f"[bestiary] threat-weighted kills: "
                      f"{', '.join(f'{k}×{v:.2f}' for k, v in mults.items())}")
        except Exception:
            pass
    if cfg.campaign:
        first_map = cfg.maps[0]
        env_fns = [
            make_campaign_env(
                cfg,
                first_map,
                rank,
                rewards=rewards,
                window_visible=cfg.render,
                memory_dir=cfg.memory_dir if cfg.memory_enabled else None,
            )
            for rank in range(cfg.n_envs)
        ]
    else:
        env_fns = [
            make_doom_env(
                cfg.scenario,
                cfg.frame_skip,
                cfg.resolution,
                cfg.seed,
                rank,
                window_visible=cfg.render,
                rewards=rewards,
            )
            for rank in range(cfg.n_envs)
        ]
    # DummyVecEnv (in-process) when rendering OR running a single env — spawning a subprocess
    # worker for one env is pointless and, under memory pressure, a worker that gets OOM-killed
    # breaks the parent's pipe (silent exit 1). SubprocVecEnv only earns its keep with N>1.
    use_dummy = cfg.render or cfg.n_envs == 1
    venv = DummyVecEnv(env_fns) if use_dummy else SubprocVecEnv(env_fns)
    venv = VecMonitor(venv)  # aggregated episode stats
    venv = VecFrameStack(venv, n_stack=cfg.frame_stack)  # stack frames (motion)
    if normalize:
        # Normalise the REWARD only (norm_obs=False — images stay raw uint8, matching eval).
        # A running return std keeps the heavily-shaped reward in a stable range for the value
        # function. clip_reward guards against shaping spikes. gamma must match PPO's.
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=cfg.gamma)
    return venv


def main() -> None:
    args = parse_args()
    cfg = apply_args(Config(), args)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Ollama is OPTIONAL: if missing, training continues and notes come out factual.
    if cfg.docs_enabled:
        _probe_ollama(cfg)
    else:
        print("[docs] disabled — pure training, no LLM calls.")
    if cfg.render:
        print("[render] Doom window enabled (1 env).")

    # Discover button names (action-distribution labels) without booting training.
    if cfg.campaign:
        meta = campaign_metadata(cfg.wad_path, cfg.maps[0], strafe=cfg.strafe)
        button_names = meta["button_names"]
        print(
            f"Campaign | WAD: {os.path.basename(cfg.wad_path)} | "
            f"maps: {list(cfg.maps)} | {cfg.steps_per_map} steps/map | "
            f"actions: {meta['num_actions']} {button_names}"
        )
    else:
        meta = probe_env_metadata(cfg.scenario, cfg.frame_skip, cfg.resolution)
        button_names = meta["button_names"]
        print(f"Scenario: {cfg.scenario} | actions: {meta['num_actions']} {button_names}")

    venv = build_vec_env(cfg, normalize=cfg.normalize_reward)
    # Include the action count so resume never loads an incompatible brain
    # (e.g., a 7-button campaign checkpoint into an 8-button one).
    task = "campaign" if cfg.campaign else cfg.scenario
    # Tag the brain family (`_lstm`, `_sp`) so a recurrent or spatial-memory brain never
    # cross-loads with an incompatible one (same guard idea as the action-count `a{N}`).
    from rl.algo import algo_class, brain_prefix, describe, policy_name
    name_prefix = brain_prefix(task, meta["num_actions"], cfg.use_lstm,
                               cfg.spatial_memory, cfg.depth_perception, cfg.automap, cfg.frame_stack, cfg.game_vars,
                               getattr(cfg, "semantic_channel", False))
    AlgoClass = algo_class(cfg.use_lstm)

    # By DEFAULT, reuse this vault's brain (don't restart from scratch).
    resume_path = _resolve_resume(cfg, args, name_prefix)
    device = _best_device()
    if resume_path:
        print(f"[brain] reusing this vault's learning: {resume_path}  [device={device}]")
        # custom_objects: replace the PICKLED schedule from the zip with the current
        # (floored) one — otherwise old brains keep their decay-to-zero schedule and
        # resumed chunks train at ~0% LR forever, no matter what config says now.
        model = AlgoClass.load(resume_path, env=venv, device=device,
                               tensorboard_log=cfg.tensorboard_log,
                               custom_objects={"learning_rate": _lr_setting(cfg)})
        reset_timesteps = False
        # Restore the reward-normalization running stats so the reward scale is continuous
        # across resumes (otherwise each chunk re-warms the return std from scratch).
        if cfg.normalize_reward and isinstance(venv, VecNormalize):
            vn = _latest_vecnorm(cfg, name_prefix)
            if vn:
                try:
                    import pickle
                    with open(vn, "rb") as f:
                        saved = pickle.load(f)
                    venv.ret_rms = saved.ret_rms
                    print(f"[norm] restored reward-normalization stats from {os.path.basename(vn)}")
                except Exception as exc:
                    print(f"[norm] couldn't restore norm stats ({exc}); re-warming fresh.")
    else:
        if args.fresh:
            print("[brain] --fresh: starting from ZERO (overwrites this vault's brain).")
        else:
            print(f"[brain] no brain in this vault ({cfg.checkpoint_dir}) — "
                  "starting from zero.")
        algo_label, pol = describe(cfg.use_lstm, cfg.game_vars)
        print(f"[brain] policy: {algo_label} / {pol}")
        model = AlgoClass(
            policy=policy_name(cfg.use_lstm, cfg.game_vars),
            env=venv,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            ent_coef=cfg.ent_coef,
            learning_rate=_lr_setting(cfg),
            clip_range=cfg.clip_range,
            seed=cfg.seed,
            device=device,
            tensorboard_log=cfg.tensorboard_log,
            verbose=1,
        )
        print(f"[brain] compute: {device}")
        reset_timesteps = True

    # `--timesteps` is ALWAYS the number of steps to train NOW. On resume, SB3 itself adds
    # the existing count (with reset_num_timesteps=False it does `total += num_timesteps`),
    # so we pass cfg.total_timesteps directly — passing num+total here double-counted and
    # trained ~2x too long. `learn_total` is just for the human-readable target message.
    if resume_path:
        learn_total = model.num_timesteps + cfg.total_timesteps
        print(f"[brain] already has {model.num_timesteps:,} steps — training "
              f"+{cfg.total_timesteps:,} (target {learn_total:,}).")

    callbacks = []
    # Boot-timing probe: a ~70-minute pre-stepping stall was observed once on a campaign
    # chunk (cumulative fps read 16 while the marginal rate was ~440 steps/s) and could
    # not be diagnosed after the fact. This prints two wall-clock marks — training_start
    # and the first env step — so the next stall shows WHERE the time went.
    from rl.callbacks import BootTimingCallback
    callbacks.append(BootTimingCallback())
    # In campaign mode, the curriculum switches maps by timesteps. Closed loop:
    # weight each map's budget by past deaths AND under-exploration there (from memory)
    # -> the agent trains MORE where it died MORE or explored LESS. No memory = uniform.
    if cfg.campaign and len(cfg.maps) > 1:
        weights = None
        if cfg.memory_enabled:
            from rl.campaign_callbacks import combined_map_weights

            events = MemoryStore.read_events(cfg.memory_dir)
            if events:
                weights = combined_map_weights(events, list(cfg.maps))
                focus = ", ".join(f"{m}×{w:.2f}" for m, w in weights.items())
                print(f"[curriculum] memory-weighted focus (more deaths/less explored = more steps): {focus}")
        callbacks.append(
            MapCurriculumCallback(
                maps=list(cfg.maps),
                steps_per_map=cfg.steps_per_map,
                loop_maps=cfg.loop_maps,
                weights=weights,
            )
        )
    log_path = log_path_for(cfg.pending_dir, cfg.run_name)
    doc_cb = None
    if cfg.docs_enabled:
        tracker = StatsTracker(button_names=button_names)
        snap_log = SnapshotLog(log_path)
        # Sidecar metadata for post-processing (writer.process_run).
        write_meta(
            meta_path_for(cfg.pending_dir, cfg.run_name),
            {
                "run_name": cfg.run_name,
                "scenario": cfg.scenario,
                "campaign": cfg.campaign,
                "maps": list(cfg.maps),
                "total_timesteps": cfg.total_timesteps,
                "button_names": button_names,
            },
        )
        doc_cb = DoomDocumentationCallback(
            tracker=tracker,
            log=snap_log,
            write_every_steps=cfg.write_every_steps,
            novelty_threshold=cfg.novelty_threshold,
        )
        callbacks.append(doc_cb)

    # Persistent memory (Phase 1): record episode-end events across runs.
    if cfg.memory_enabled:
        callbacks.append(
            MemoryRecorderCallback(MemoryStore(cfg.memory_dir, run_name=cfg.run_name))
        )
        # Per-map explored heatmap that persists across runs (campaign only — the
        # geometry/coverage of full maps is what's worth remembering across layouts).
        if cfg.campaign:
            from rl.coverage_callback import CoverageMemoryCallback
            from rl.enemy_callback import EnemyMemoryCallback
            from writer.bestiary import BestiaryStore
            from writer.coverage_store import CoverageStore

            callbacks.append(CoverageMemoryCallback(CoverageStore(cfg.memory_dir)))
            callbacks.append(EnemyMemoryCallback(BestiaryStore(cfg.memory_dir)))

    # Feedback loop: Obsidian (00-index/control.md) controls training live.
    if cfg.control_enabled:
        control_path = os.path.join(cfg.vault_path, cfg.dir_index, "control.md")
        callbacks.append(
            ControlCallback(
                control_path=control_path,
                every_steps=cfg.control_every_steps,
                doc_callback=doc_cb,
            )
        )
    callbacks.append(
        CheckpointCallback(
            save_freq=max(cfg.write_every_steps // cfg.n_envs, 1),
            save_path=cfg.checkpoint_dir,
            name_prefix=name_prefix,
            save_vecnormalize=cfg.normalize_reward,  # persist reward-norm stats with the brain
        )
    )

    # progress_bar exige tqdm+rich; se faltarem, segue sem barra em vez de quebrar.
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401

        use_bar = True
    except ImportError:
        use_bar = False
        print("[info] tqdm/rich missing — continuing without a progress bar.")

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,  # SB3 adds the existing count on resume
            callback=callbacks,
            progress_bar=use_bar,
            reset_num_timesteps=reset_timesteps,
        )
    finally:
        model.save(os.path.join(cfg.checkpoint_dir, f"{name_prefix}_final"))
        venv.close()

    # POST-TRAINING: only now do we call the LLM, in batch. The PPO loop is already
    # done, so generating notes no longer blocks anything (the freeze fix).
    if cfg.docs_enabled:
        print("\n[docs] training finished — generating Obsidian notes (batch)...")
        from writer.process_run import process_run

        process_run(cfg, button_names, log_path)


if __name__ == "__main__":
    main()
