"""Deterministic evaluation of a saved brain — the rigorous way to test performance.

Training metrics are noisy (exploration + reward shaping). This loads a checkpoint and
runs N episodes with `deterministic=True` (no exploration), reporting clean numbers:
mean reward, shooting accuracy, kills/episode, success rate. Use it to:
  - measure a brain's real performance, and
  - A/B two brains (same task, change one thing) by comparing their eval numbers.

    python -m rl.eval                      # evaluate this vault's brain (20 episodes)
    python -m rl.eval --episodes 50
    python -m rl.eval --path ./checkpoints/ppo_defend_the_center_a3_final.zip
"""
import argparse
import time
from typing import Optional

import numpy as np

from config import Config
from doom.campaign import campaign_metadata
from doom.env import probe_env_metadata
from instrumentation.stats_tracker import StatsTracker
from rl.train import _latest_checkpoint, build_vec_env


def _tempered_actions(model, obs, temperature: float) -> np.ndarray:
    """Sample discrete actions from the policy with its logits scaled by 1/temperature.

    temperature → 0  ≈ argmax (deterministic),  1.0 = normal sampling,  >1 = flatter/random.
    A value like 0.5 sharpens toward the best actions while keeping enough stochasticity to
    avoid the argmax-collapse where the single most-probable action is a bad one."""
    import torch
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)
        logits = dist.distribution.logits          # Categorical (discrete action space)
        tempered = torch.distributions.Categorical(logits=logits / max(temperature, 1e-6))
        actions = tempered.sample()
    return actions.cpu().numpy()


def evaluate(cfg: Config, path: str, button_names: list, episodes: int = 20,
             deterministic: bool = True, temperature: Optional[float] = None,
             overlay: bool = False, recall: bool = False,
             recall_threshold: float = 0.92, recall_skip_noop: bool = False) -> dict:
    """Run `episodes` and return a metrics summary.

    `deterministic=True` (default) takes the argmax action — the honest measure of what the
    brain has *committed* to. `deterministic=False` samples the policy at temperature 1.0.
    `temperature=τ` (overrides both) samples with logits scaled by 1/τ — a tunable middle
    ground that lets a brain whose argmax collapsed still act on its learned distribution
    (e.g. τ=0.5 explores without freezing). Feed-forward policies only.

    `recall=True` enables demo retrieval: at each step, if the current view closely matches a
    frame from the human demos (cosine ≥ recall_threshold), replay the human's action there
    instead of the policy's — "search memory for the best action". Defers to the policy when
    no close match (the confidence gate)."""
    venv = build_vec_env(cfg)  # n_envs forced to 1 by the caller
    from rl.algo import algo_class_from_path
    import os
    AlgoClass = algo_class_from_path(path)
    use_lstm = "_lstm" in os.path.basename(path).lower()
    if temperature is not None and use_lstm:
        print("[eval] temperature sampling is feed-forward only; ignoring for the LSTM brain.")
        temperature = None
    model = AlgoClass.load(path, env=venv)
    # Tempered sampling needs a STOCHASTIC policy (PPO's Categorical via get_distribution).
    # QR-DQN is value-based — its "policy" is argmax over Q-values, with no action
    # distribution — so temperature can't apply. Fall back to deterministic (the honest
    # measure for DQN: it has no argmax-collapse pathology that temperature would dodge).
    if temperature is not None and not hasattr(model.policy, "get_distribution"):
        print("[eval] temperature sampling needs a stochastic policy; this brain is "
              "value-based (QR-DQN) — using deterministic argmax instead.")
        temperature = None
    tracker = StatsTracker(button_names=button_names)
    # When rendering, throttle to ~real time so the window is actually watchable
    # (otherwise ViZDoom blasts through hundreds of fps and the episodes flash by).
    step_delay = (cfg.frame_skip / 35.0) if cfg.render else 0.0

    # Demo retrieval ("search memory for the best action"): index the human demos and, when the
    # live view closely matches a demo frame, replay the human's action there.
    retriever = None
    recall_hits = recall_steps = 0
    # The pixel channel of the MOST RECENT stacked frame (channels-last [N,H,W,C]; channels
    # per frame = C // frame_stack; pixels are channel 0 of each frame group).
    _pix_ch = None
    if recall:
        from rl.demo_retrieval import DemoRetriever
        from rl.frame_encoder import load_encoder_if_present
        demos_dir = os.path.join(cfg.memory_dir, "demos")
        encoder = load_encoder_if_present(cfg.memory_dir)  # learned embedding if trained
        retriever = DemoRetriever(demos_dir, skip_noop=recall_skip_noop, encoder=encoder)
        if len(retriever) == 0:
            print(f"[eval] --recall: no usable demos in {demos_dir}; ignoring recall.")
            retriever = None
        else:
            kind = "learned-embedding" if encoder is not None else "pixel-descriptor"
            print(f"[eval] recall ON: {len(retriever)} demo frames, {kind} "
                  f"(threshold {recall_threshold:.2f}"
                  f"{', skip no-op' if recall_skip_noop else ''})")

    # Overlay window (watch --overlay): renders HUD + minimap in a cv2 window.
    # ViZDoom's own window already opens when cfg.render=True; the overlay is a
    # SECOND annotated window that shows the agent's obs + health/ammo bars.
    _win = None
    if overlay and cfg.render:
        try:
            import cv2 as _cv2
            _win = "HeLLMind — overlay"
            _cv2.namedWindow(_win, _cv2.WINDOW_NORMAL)
            _cv2.resizeWindow(_win, 420, 440)
        except ImportError:
            _win = None
            print("[eval] install opencv-python for the overlay (pip install opencv-python)")

    obs = venv.reset()
    done_count = 0
    # Recurrent-safe loop: carry the LSTM hidden state and flag episode boundaries.
    lstm_states = None
    episode_starts = np.ones((venv.num_envs,), dtype=bool)
    while done_count < episodes:
        if temperature is not None:
            action = _tempered_actions(model, obs, temperature)
        else:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts,
                deterministic=deterministic)
        # Demo recall: override the policy with the human's action in the most similar demo
        # frame, but only when the match is confident (else keep the policy's action).
        if retriever is not None:
            img = obs["image"] if isinstance(obs, dict) else obs
            img = np.asarray(img)
            if _pix_ch is None:                          # resolve the newest pixel channel once
                c = img.shape[-1]
                _pix_ch = c - max(1, c // max(1, cfg.frame_stack))
            frame = img[0, :, :, _pix_ch]                # env 0, newest pixel frame (84x84)
            mem_action, _sim = retriever.retrieve(frame, recall_threshold)
            recall_steps += 1
            if mem_action is not None:
                action = np.asarray(action)
                action[0] = mem_action
                recall_hits += 1
        obs, _rewards, dones, infos = venv.step(action)
        episode_starts = dones
        tracker.update(infos, np.asarray(action))
        done_count += sum(1 for i in infos if i.get("episode"))
        # Draw the overlay window every step
        if _win is not None:
            try:
                import cv2 as _cv2
                from doom.overlay import (draw_hud, draw_object_boxes, draw_door_map,
                                          draw_semantic_panel)
                img_obs = np.asarray(obs["image"] if isinstance(obs, dict) else obs)
                frame = img_obs[0, :, :, 0]   # first channel, env 0
                bgr = _cv2.cvtColor(
                    _cv2.resize(frame, (420, 420), interpolation=_cv2.INTER_NEAREST),
                    _cv2.COLOR_GRAY2BGR)
                doom_i = (infos[0].get("doom") or {}) if infos else {}
                # Squares around EVERY object the agent sees (the on-screen detector).
                draw_object_boxes(bgr, doom_i.get("objects"), 420, 420)
                # Door minimap (top-right): where the doors are + where the agent is headed.
                draw_door_map(bgr, doom_i.get("navmap"))
                # Semantic channel panel (bottom-right): what the NETWORK sees, if that channel
                # is in the obs (last channel of the newest stacked frame).
                if getattr(cfg, "semantic_channel", False):
                    draw_semantic_panel(bgr, img_obs[0, :, :, img_obs.shape[-1] - 1])
                lvl = doom_i.get("levels", {})
                if lvl:
                    draw_hud(bgr,
                             min(1.0, float(lvl.get("health", 0)) / 100.0),
                             min(1.0, float(lvl.get("ammo2", 0)) / 50.0))
                _cv2.imshow(_win, bgr)
                if _cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except Exception:
                pass
        if step_delay:
            time.sleep(step_delay)
    if _win is not None:
        try:
            import cv2 as _cv2; _cv2.destroyAllWindows()
        except Exception:
            pass
    venv.close()
    snap = tracker.snapshot(0)
    if retriever is not None and recall_steps > 0:
        snap["recall_hit_rate"] = recall_hits / recall_steps
        snap["recall_hits"] = recall_hits
    return snap


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministically evaluate a saved brain.")
    p.add_argument("--episodes", type=int, default=20, help="Episodes to run.")
    p.add_argument("--path", default=None, help="Checkpoint .zip (default: vault's latest).")
    p.add_argument("--render", action="store_true", help="Show the Doom window.")
    p.add_argument("--json", action="store_true",
                   help="Also print a one-line JSON of the key metrics (for the supervisor).")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample the policy instead of argmax — reveals what an unconverged "
                        "brain has learned but can't yet argmax (e.g. fights when sampled).")
    p.add_argument("--temperature", type=float, default=None,
                   help="Tempered sampling (e.g. 0.5): scales logits by 1/T. Sharpens toward "
                        "the best actions but avoids the argmax-collapse. Overrides --stochastic.")
    p.add_argument("--overlay", action="store_true",
                   help="Show HUD + minimap overlay in a separate cv2 window (needs opencv-python).")
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"],
                   help="Which brain family to evaluate: ppo (default) or dqn (QR-DQN). "
                        "Selects the checkpoint prefix when --path is not given.")
    p.add_argument("--recall", action="store_true",
                   help="Demo retrieval: replay the human's action from the most similar demo "
                        "frame when the live view matches closely (search memory for the best "
                        "action). Defers to the policy when there's no confident match.")
    p.add_argument("--recall-threshold", type=float, default=0.92,
                   help="Cosine-similarity gate for --recall (default 0.92). Higher = only very "
                        "close matches replay the human action.")
    p.add_argument("--recall-skip-noop", action="store_true",
                   help="With --recall, ignore the human's idle (no-op) frames so retrieval "
                        "suggests an active move instead of freezing.")
    p.add_argument("--html", nargs="?", const="reports/eval_report.html", default=None,
                   help="Write a full HTML report (metrics + charts + formulas + recommendations) "
                        "after the eval. Optional path (default reports/eval_report.html).")
    p.add_argument("--seed", type=int, default=None,
                   help="Pin EVERY rng (env layout + torch/numpy action sampling) so two evals "
                        "are comparable. Without it, tempered sampling draws from an unseeded "
                        "torch RNG and A/B numbers carry sampling noise.")
    args = p.parse_args()

    cfg = Config()
    cfg.n_envs = 1            # single env for a clean, reproducible eval
    cfg.docs_enabled = False  # no LLM/notes during eval
    cfg.memory_enabled = False
    if args.seed is not None:
        import random
        import numpy as _np
        import torch as _torch
        cfg.seed = args.seed          # env reset seed (make_campaign_env: seed + rank)
        random.seed(args.seed)
        _np.random.seed(args.seed)
        _torch.manual_seed(args.seed)  # tempered/stochastic action sampling
        print(f"[eval] seed pinned: {args.seed} (env + torch + numpy)")
    if args.render:
        cfg.render = True

    from rl.algo import brain_prefix
    if cfg.campaign:
        meta = campaign_metadata(cfg.wad_path, cfg.maps[0], strafe=cfg.strafe)
    else:
        meta = probe_env_metadata(cfg.scenario, cfg.frame_skip, cfg.resolution)
    # The brain name differs by algorithm: QR-DQN uses the qrdqn_ prefix (with the same obs
    # tags), PPO/RecurrentPPO use brain_prefix. Picking the wrong one means eval looks for a
    # checkpoint that doesn't exist → "No checkpoint found" (the DQN auto-loop's eval failure).
    if args.algo == "dqn":
        from rl.train_dqn import _dqn_prefix
        name_prefix = _dqn_prefix(meta["num_actions"], cfg.game_vars, cfg)
    else:
        task = "campaign" if cfg.campaign else cfg.scenario
        name_prefix = brain_prefix(task, meta["num_actions"], cfg.use_lstm,
                                   cfg.spatial_memory, cfg.depth_perception, cfg.automap,
                                   cfg.frame_stack, cfg.game_vars,
                                   getattr(cfg, "semantic_channel", False))
    button_names = meta["button_names"]

    path = args.path or _latest_checkpoint(cfg, name_prefix)
    if not path:
        raise SystemExit(f"No checkpoint found for '{name_prefix}' in {cfg.checkpoint_dir}. "
                         "Train first, or pass --path.")
    if args.temperature is not None:
        mode = f"tempered sampling (T={args.temperature})"
    elif args.stochastic:
        mode = "stochastic (sampled)"
    else:
        mode = "deterministic (argmax)"
    print(f"[eval] {path} | {args.episodes} {mode} episodes")

    s = evaluate(cfg, path, button_names, args.episodes,
                 deterministic=not args.stochastic, temperature=args.temperature,
                 overlay=getattr(args, "overlay", False), recall=args.recall,
                 recall_threshold=args.recall_threshold,
                 recall_skip_noop=args.recall_skip_noop)
    print("\n== Evaluation ==")
    print(f"  episodes:        {int(s['episodes'])}")
    print(f"  RAW reward/ep:   {s['mean_base_reward']:.2f}   (native scenario, fair for A/B)")
    print(f"  shaped reward/ep:{s['mean_reward']:.2f}   (includes reward shaping)")
    print(f"  shooting acc.:   {s['shooting_accuracy']:.0%}")
    print(f"  kills/episode:   {s['kills_per_episode']:.2f}")
    print(f"  success rate:    {s['success_rate']:.0%}")
    print(f"  exit rate:       {s.get('exit_rate', 0.0):.0%}   (reached the level end)")
    if s.get("exit_progress", 0.0) > 0:
        print(f"  exit progress:   {s.get('exit_progress', 0.0):.0%}   "
              f"(how close it got to the known exit — fairer than the binary rate)")
    cov = s.get("map_coverage", {}) or {}
    print(f"  map explored:    {cov.get('explored_fraction', 0.0):.0%}   "
          f"({int(cov.get('cells_visited', 0))} cells)")
    if s.get("terminals"):
        print(f"  episode endings: {s['terminals']}")
    print(f"  mean ep length:  {s['mean_episode_length']:.0f} steps")
    # "What happened this run" — grouped panels so you can see WHAT to adjust (per episode).
    n_eps = max(int(s.get("episodes", 1)), 1)
    dist = s.get("action_distribution", {}) or {}

    def _share(*tokens):
        return sum(v for k, v in dist.items() if any(t in k for t in tokens))

    print("\n  -- AIM (did it learn to shoot well?) --")
    print(f"  accuracy:        {s.get('shooting_accuracy', 0.0):.0%}   "
          f"(landed {s.get('shots_hit', 0.0)/n_eps:.1f} of {s.get('shots_fired_per_episode', 0.0):.1f} shots/ep)")
    print(f"  shots per kill:  {s.get('shots_per_kill', 0.0):.1f}   (lower = better aim/discipline)")
    print(f"  aim offset:      {s.get('aim_offset', 0.0):.2f}   (0=enemy dead-centre, 1=screen edge)")
    print(f"  wasted shots:    {s.get('wasted_shot_rate', 0.0):.0%}   (fired with NO enemy on screen)")
    print(f"  kill conversion: {s.get('kill_conversion', 0.0):.0%}   (enemies killed of those seen)")
    if s.get("reaction_ticks"):
        print(f"  reaction:        {s.get('reaction_ticks', 0.0):.0f} ticks (enemy seen → first shot)")
    if s.get("nearest_enemy_dist"):
        print(f"  enemy distance:  {s.get('nearest_enemy_dist', 0.0):.0f} units avg "
              f"(lower = lets them close in)")

    print("  -- MOVEMENT --")
    print(f"  explored:        {(s.get('map_coverage', {}) or {}).get('explored_fraction', 0.0):.0%}   "
          f"({s.get('distance_per_episode', 0.0):.0f} units/ep, frontier {s.get('frontier_reach', 0.0):.0f})")
    print(f"  idle/stuck:      {s.get('idle_rate', 0.0):.0%} of steps   "
          f"revisit {s.get('revisit_rate', 0.0):.0%} (circling)")
    print(f"  style:           fwd {_share('FWD'):.0%} · turn {_share('TL','TR','TURN'):.0%} · "
          f"strafe {_share('SL','SR','MOVE_LEFT','MOVE_RIGHT'):.0%} · back {_share('BACK'):.0%}")

    print("  -- WEAPONS --")
    print(f"  distinct used:   {s.get('distinct_weapons_used', 0.0):.0f}   "
          f"(switches {s.get('weapon_switches_per_episode', 0.0):.1f}/ep, "
          f"best-gun {s.get('best_weapon_fraction', 0.0):.0%} of the time)")
    # Per-weapon detail: WHICH gun, HOW LONG (% of time wielded) and HOW WELL (accuracy).
    wu = s.get("weapons_used", {}) or {}            # slot -> fraction of time
    abw = s.get("accuracy_by_weapon", {}) or {}     # slot -> hit rate
    if wu:
        for slot in sorted(wu, key=lambda k: -wu[k]):   # most-used first
            acc = abw.get(slot)
            acc_s = f", {acc:.0%} acc" if acc is not None else ""
            print(f"    {slot.replace('slot_', 'weapon ')}: {wu[slot]:.0%} of time{acc_s}")

    print("  -- PERCEPTION (what it identified) --")
    seen = s.get("objects_seen_per_episode", {}) or {}
    if seen:
        order = ["enemy", "weapon", "health", "ammo", "key", "item"]
        parts = [f"{c} {seen[c]:.1f}" for c in order if c in seen]
        print(f"  objects seen/ep: {' · '.join(parts)}")
    print(f"  pickup conv.:    {s.get('pickup_conversion', 0.0):.0%}   (items grabbed of those seen)")
    print(f"  doors reached:   {s.get('doors_reached_per_episode', 0.0):.1f}/ep   "
          f"exit progress {s.get('exit_progress', 0.0):.0%}")

    print("  -- SURVIVAL & POLICY --")
    print(f"  survival:        {s.get('hits_taken_per_episode', 0.0):.1f} hits ({s.get('damage_taken', 0.0)/n_eps:.0f} HP), "
          f"{s.get('heals_consumed', 0.0)/n_eps:.1f} heals, {s.get('low_health_fraction', 0.0):.0%} time low HP, "
          f"{s.get('out_of_ammo_fraction', 0.0):.0%} dry")
    rb = s.get("reward_breakdown", {}) or {}
    if rb:
        order = ["combat", "engage", "explore", "move", "damage", "base"]
        print("  reward from:     " + " · ".join(f"{k} {rb[k]:+.0%}"
                                                 for k in order if k in rb)
              + "   (what it optimises)")
    print(f"  decisiveness:    {s.get('action_entropy_normalized', 0.0):.2f} "
          f"action-entropy (0=fixed, 1=random)")
    if "recall_hit_rate" in s:
        print(f"  demo recall:     {s['recall_hit_rate']:.0%} of steps replayed a human action "
              f"({s['recall_hits']} steps from memory)")

    # Export the full metrics to Prometheus (push-gateway / textfile) if configured — for
    # Grafana time-series of the agent improving across runs. No-op unless the env vars are set.
    try:
        from instrumentation.prometheus_exporter import export_metrics
        export_metrics(s, job="hellmind_eval")
    except Exception as _e:
        print(f"[prometheus] export skipped: {_e}")

    # Full HTML report (metrics + charts + formulas + recommendations) if requested.
    if args.html:
        try:
            import os as _os
            from writer.html_report import write_report
            report_path = write_report(
                s, args.html,
                meta={"map": cfg.maps[0] if cfg.campaign else cfg.scenario,
                      "brain": _os.path.basename(path)})
            print(f"\n[report] HTML written -> {report_path}")
        except Exception as _e:
            print(f"[report] HTML failed: {_e}")

    if args.json:
        import json
        cov = s.get("map_coverage", {}) or {}
        n_eps = max(s.get("episodes", 1), 1)
        terminals = s.get("terminals", {})
        metrics = {
            "kills_per_episode": float(s["kills_per_episode"]),
            "shooting_accuracy": float(s["shooting_accuracy"]),
            "success_rate": float(s["success_rate"]),
            "exit_rate": float(s.get("exit_rate", 0.0)),
            "exit_progress": float(s.get("exit_progress", 0.0)),  # dense: how close (euclidean)
            # Geodesic route metrics (None-safe: tracker means are None when unmeasured).
            "route_progress": float(s.get("route_progress") or 0.0),
            "route_progress_best": float(s.get("route_progress_best") or 0.0),
            "death_route_dist": float(s.get("death_route_dist") or 0.0),
            "timeout_rate": float(terminals.get("timeout", 0)) / n_eps,
            "death_rate": float(terminals.get("death", 0)) / n_eps,
            "explored_fraction": float(cov.get("explored_fraction", 0.0)),
            "cells_visited": float(cov.get("cells_visited", 0.0)),
            "enemies_seen_per_episode": float(s.get("enemies_seen_per_episode", 0.0)),
            "hits_taken_per_episode": float(s.get("hits_taken_per_episode", 0.0)),
            "heals_consumed": float(s.get("heals_consumed", 0.0)),
            # Skill-curriculum scoreboard: aim quality + spray + conversion + circling + reward mix.
            "aim_offset": float(s.get("aim_offset", 0.0)),
            "wasted_shot_rate": float(s.get("wasted_shot_rate", 0.0)),
            "kill_conversion": float(s.get("kill_conversion", 0.0)),
            "revisit_rate": float(s.get("revisit_rate", 0.0)),
            "reward_breakdown": s.get("reward_breakdown", {}),
            "mean_base_reward": float(s["mean_base_reward"]),
            "mean_episode_length": float(s["mean_episode_length"]),
            # Combat vs exploration regime — so the coach tunes each one separately.
            "combat_fraction": float(s.get("combat_fraction", 0.0)),
            "combat_engagement": float(s.get("combat_engagement", 0.0)),
            "combat_accuracy": float(s.get("combat_accuracy", 0.0)),
        }
        print("METRICS_JSON " + json.dumps(metrics))


if __name__ == "__main__":
    main()
