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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
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
    """The most recent brain for this task in the vault (prefers _final)."""
    final = os.path.join(cfg.checkpoint_dir, f"{name_prefix}_final.zip")
    if os.path.exists(final):
        return final
    candidates = sorted(
        glob.glob(os.path.join(cfg.checkpoint_dir, f"{name_prefix}*.zip")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def _resolve_resume(cfg: Config, args: argparse.Namespace, name_prefix: str) -> Optional[str]:
    """By DEFAULT reuse the vault's brain. --fresh starts from zero; --resume PATH
    uses a specific file."""
    if args.fresh:
        return None
    if args.resume and args.resume != "auto":
        path = args.resume if args.resume.endswith(".zip") else args.resume + ".zip"
        return path if os.path.exists(path) else None
    return _latest_checkpoint(cfg, name_prefix)  # default: continue where it stopped


def build_vec_env(cfg: Config):
    rewards = cfg.reward_weights()
    if cfg.campaign:
        first_map = cfg.maps[0]
        env_fns = [
            make_campaign_env(
                cfg.wad_path,
                first_map,
                cfg.frame_skip,
                cfg.resolution,
                cfg.episode_timeout,
                cfg.kills_to_clear,
                cfg.seed,
                rank,
                window_visible=cfg.render,
                rewards=rewards,
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
    # Render needs a window in the main process -> DummyVecEnv (no subprocess).
    venv = DummyVecEnv(env_fns) if cfg.render else SubprocVecEnv(env_fns)
    venv = VecMonitor(venv)  # aggregated episode stats
    venv = VecFrameStack(venv, n_stack=cfg.frame_stack)  # stack frames (motion)
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
        meta = campaign_metadata(cfg.wad_path, cfg.maps[0])
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

    venv = build_vec_env(cfg)
    # Include the action count so resume never loads an incompatible brain
    # (e.g., a 7-button campaign checkpoint into an 8-button one).
    task = "campaign" if cfg.campaign else cfg.scenario
    name_prefix = f"ppo_{task}_a{meta['num_actions']}"

    # By DEFAULT, reuse this vault's brain (don't restart from scratch).
    resume_path = _resolve_resume(cfg, args, name_prefix)
    if resume_path:
        print(f"[brain] reusing this vault's learning: {resume_path}")
        model = PPO.load(resume_path, env=venv, tensorboard_log=cfg.tensorboard_log)
        reset_timesteps = False
    else:
        if args.fresh:
            print("[brain] --fresh: starting from ZERO (overwrites this vault's brain).")
        else:
            print(f"[brain] no brain in this vault ({cfg.checkpoint_dir}) — "
                  "starting from zero.")
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            ent_coef=cfg.ent_coef,
            learning_rate=cfg.learning_rate,
            clip_range=cfg.clip_range,
            seed=cfg.seed,
            tensorboard_log=cfg.tensorboard_log,
            verbose=1,
        )
        reset_timesteps = True

    # When reusing the brain, `--timesteps` is ADDITIONAL (train X more); otherwise
    # SB3 would treat it as an absolute target and do nothing if it's already met.
    if resume_path:
        learn_total = model.num_timesteps + cfg.total_timesteps
        print(f"[brain] already has {model.num_timesteps:,} steps — training "
              f"+{cfg.total_timesteps:,} (target {learn_total:,}).")
    else:
        learn_total = cfg.total_timesteps

    callbacks = []
    # In campaign mode, the curriculum switches maps by timesteps. Closed loop:
    # weight each map's budget by past deaths there (from memory) -> the agent
    # trains MORE where it died MORE. No memory yet = uniform (previous behavior).
    if cfg.campaign and len(cfg.maps) > 1:
        weights = None
        if cfg.memory_enabled:
            from rl.campaign_callbacks import map_step_weights

            events = MemoryStore.read_events(cfg.memory_dir)
            if events:
                weights = map_step_weights(events, list(cfg.maps))
                focus = ", ".join(f"{m}×{w:.2f}" for m, w in weights.items())
                print(f"[curriculum] memory-weighted focus (more deaths = more steps): {focus}")
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
            total_timesteps=learn_total,
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
