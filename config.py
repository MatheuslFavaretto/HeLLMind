"""Central project configuration. Reads from the environment (.env) with sane defaults."""
import os
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: str) -> int:
    """Parse an int-typed env var, tolerating float-like strings ('2100.0').
    The auto-loop / memory_policy can emit floats for int knobs (e.g. EPISODE_TIMEOUT);
    int('2100.0') would crash, so route every int knob through float() first."""
    raw = os.getenv(name, default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(float(default))


@dataclass
class Config:
    # ---------- Doom environment ----------
    scenario: str = os.getenv("DOOM_SCENARIO", "defend_the_center")
    frame_skip: int = 4
    # Stacked frames (motion). Tunable because each extra obs channel (spatial/depth/automap)
    # multiplies by frame_stack — so with all perception channels on, drop to 2 to fit memory
    # (4 base channels × 2 = 8, same footprint as spatial-only × 4). Tagged in the brain name.
    frame_stack: int = _int_env("FRAME_STACK", "2")
    resolution: Tuple[int, int] = (84, 84)  # (width, height)

    # ---------- CAMPAIGN mode (full WAD maps, in order) ----------
    # Enable campaign mode (train full maps and advance to the next).
    campaign: bool = os.getenv("CAMPAIGN", "1") in ("1", "true", "True")
    # WAD with the maps. Default: bundled freedoom2.wad (free).
    # For the original Doom 1 maps, point to your doom.wad.
    wad_path: str = os.getenv("WAD_PATH", "")
    # Map list, in order. freedoom2 uses MAP01..; doom.wad uses E1M1..
    maps: Tuple[str, ...] = tuple(
        os.getenv("MAPS", "MAP01,MAP02,MAP03,MAP04,MAP05").split(",")
    )
    steps_per_map: int = int(os.getenv("STEPS_PER_MAP", "200000"))
    loop_maps: bool = os.getenv("LOOP_MAPS", "0") in ("1", "true", "True")
    kills_to_clear: int = _int_env("KILLS_TO_CLEAR", "5")
    episode_timeout: int = _int_env("EPISODE_TIMEOUT", "2800")  # ticks

    # ---------- PPO training ----------
    total_timesteps: int = int(os.getenv("TOTAL_TIMESTEPS", "2000000"))
    n_envs: int = _int_env("N_ENVS", "8")
    # Rollout length per env. CRITICAL vs episode length: if n_steps ≈ episode length, each
    # rollout holds ~1 (often truncated) episode → poor GAE advantage estimates and slow
    # learning. Keep n_steps a few× SHORTER than the episode so several episodes fit per
    # rollout. Smaller n_steps also shrinks the obs rollout buffer (the memory hog with
    # spatial 8-channel obs), which is what lets more envs fit on a 16GB machine.
    n_steps: int = _int_env("N_STEPS", "1024")
    batch_size: int = int(os.getenv("BATCH_SIZE", "256"))
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Entropy bonus: keeps the policy from collapsing its argmax onto one (often bad) action.
    # Tunable because a too-low value causes the stochastic policy to explore/fight while the
    # DETERMINISTIC (argmax) policy freezes — the exact gap measured on this agent.
    ent_coef: float = float(os.getenv("ENT_COEF", "0.03"))
    learning_rate: float = 2.5e-4
    # Linearly decay the learning rate to 0 over each training call (a standard PPO practice
    # that stabilises late training). On the chunked auto loop it's a per-chunk warm-restart
    # decay. 1 = on.
    lr_schedule: bool = os.getenv("LR_SCHEDULE", "1") in ("1", "true", "True")
    # Normalise the (heavily shaped) reward with a running return std during training — helps
    # PPO's value function when shaping terms have very different scales. Obs are NOT
    # normalised (images stay raw). 1 = on.
    normalize_reward: bool = os.getenv("NORMALIZE_REWARD", "1") in ("1", "true", "True")
    # Temperature the auto loop uses to SCORE the policy (not pure argmax). This agent's
    # argmax collapses to passive while its learned distribution explores+fights, so scoring
    # argmax would optimise a frozen policy. 0 = score argmax; 0.5 = tempered (recommended).
    eval_temperature: float = float(os.getenv("EVAL_TEMPERATURE", "0.5"))
    clip_range: float = 0.2
    seed: int = int(os.getenv("SEED", "42"))  # configurable for multi-seed A/B

    # ---------- Reward shaping weights (tunable via .env; Phase 6 can suggest) ----------
    hit_reward: float = float(os.getenv("HIT_REWARD", "1.0"))
    miss_penalty: float = float(os.getenv("MISS_PENALTY", "0.25"))
    # Campaign kill bonus (per enemy killed) — the strongest combat lever. Raise it to pull
    # the agent out of the idle-to-timeout local optimum on maps where it avoids fighting.
    kill_reward: float = float(os.getenv("KILL_REWARD", "5.0"))
    damage_taken_penalty: float = float(os.getenv("DAMAGE_TAKEN_PENALTY", "0.05"))
    death_penalty: float = float(os.getenv("DEATH_PENALTY", "5.0"))
    # Anti-idle: reward movement, and make standing still cost reward (time penalty).
    move_reward: float = float(os.getenv("MOVE_REWARD", "0.002"))
    living_reward: float = float(os.getenv("LIVING_REWARD", "-0.005"))
    # Anti-circle: reward NET outward progress (new max distance from spawn). Raw move_reward
    # pays the agent to spin in circles (infinite distance); this only pays for going OUTWARD,
    # so it can't be farmed by a limit cycle. Drives directed exploration. Campaign only.
    frontier_reward: float = float(os.getenv("FRONTIER_REWARD", "0.05"))
    # ---------- Exploration & completion (autonomy goal: explore the whole map) ----------
    # Bonus the first time the agent steps on a NEW grid cell (count-based exploration).
    # Drives covering the map instead of pacing the same corridor.
    coverage_reward: float = float(os.getenv("COVERAGE_REWARD", "0.5"))
    # Grid cell size (map units) for both the coverage reward and the spatial memory.
    coverage_cell: float = float(os.getenv("COVERAGE_CELL", "96.0"))
    # Big bonus for actually reaching the level EXIT (episode ends, not dead, pre-timeout).
    # NOTE: keep this default in sync with .env (EXIT_REWARD=1000). A large sparse terminal
    # reward is high-variance across seeds (it's only ever seen if the agent stumbles onto
    # the exit) — exit_prox_scale mitigates that by shaping a dense gradient toward the
    # exit AFTER the first success memorises its position (see campaign.py).
    exit_reward: float = float(os.getenv("EXIT_REWARD", "1000.0"))
    exit_prox_scale: float = float(os.getenv("EXIT_PROX_SCALE", "0.1"))
    # Count-based weapon variety: bonus the FIRST time the agent wields a NEW weapon slot
    # in an episode (campaign only). The campaign has SELECT_NEXT_WEAPON but, without this,
    # no reason to ever switch — so it never used the weapons it picks up.
    weapon_variety_reward: float = float(os.getenv("WEAPON_VARIETY_REWARD", "0.5"))
    # Engagement reward (needs USE_LABELS): tiny bonus per step for keeping a visible enemy
    # CENTRED in view (in the crosshair). Encourages facing/approaching enemies instead of
    # wandering past them — complements hit/miss. Keep small so it can't replace killing. 0=off.
    engagement_reward: float = float(os.getenv("ENGAGEMENT_REWARD", "0.01"))
    # Combat/exploration decoupling (Arnold/ViZDoom-champion style): pursue ONE objective at a
    # time, gated by ground-truth enemy visibility (USE_LABELS). Enemy on screen -> COMBAT
    # focus (damp exploration pulls so it doesn't wander off mid-fight); screen clear ->
    # EXPLORE focus (damp the miss penalty so blind shots while navigating aren't punished).
    # factor = how hard to damp the off-mode rewards (0.25 = keep 25%). Needs USE_LABELS.
    combat_explore_split: bool = os.getenv("COMBAT_EXPLORE_SPLIT", "1") in ("1", "true", "True")
    combat_explore_factor: float = float(os.getenv("COMBAT_EXPLORE_FACTOR", "0.25"))
    # Auto-USE: press the USE button every frame so DOORS open on contact (and switches
    # activate). Without this the agent must learn the rare FWD+USE action with no reward
    # signal -> it gets stuck banging on closed doors (observed). On by default; the agent
    # still chooses where to GO, this just stops doors from being a dead end.
    auto_use: bool = os.getenv("AUTO_USE", "1") in ("1", "true", "True")
    # Automatic goal/discovery reward (needs USE_LABELS): bonus the FIRST time each episode the
    # agent SEES a new notable object (key, switch-adjacent item, weapon, powerup, a new monster
    # type) — progress-guided exploration toward objectives, not blind wandering. 0 = off.
    discovery_reward: float = float(os.getenv("DISCOVERY_REWARD", "0.5"))
    # Closed loop (bestiary -> reward): scale the kill bonus by what the agent LEARNED about
    # each monster — killing a deadlier type (higher death-rate-when-present) pays more. Uses
    # the persisted bestiary; needs one prior run to have data. Opt-in (changes the reward).
    bestiary_reward: bool = os.getenv("BESTIARY_REWARD", "1") in ("1", "true", "True")
    # Spatial memory: feed the agent a 2nd obs channel of where it has already been
    # (so it can ACT on its own memory, not just be rewarded for it).
    spatial_memory: bool = os.getenv("SPATIAL_MEMORY", "1") in ("1", "true", "True")
    # Depth perception: feed ViZDoom's depth buffer as an extra obs channel. Gives the CNN
    # explicit 3D structure (how far each pixel is) — a strong, well-established navigation
    # signal in 3D FPS agents (UNREAL/Arnold). Changes the obs shape, so it needs --fresh.
    depth_perception: bool = os.getenv("DEPTH_PERCEPTION", "1") in ("1", "true", "True")
    # Strafe: add sideways-movement actions (dodging + navigation). Changes the action count
    # (so the brain name's a{N} differs), so switching it needs --fresh.
    strafe: bool = os.getenv("STRAFE", "1") in ("1", "true", "True")
    # Automap: feed ViZDoom's native top-down automap (explored layout + walls) as an extra
    # obs channel — an allocentric map the agent can navigate by (stronger than the hand-built
    # spatial-memory grid). Changes the obs shape, so it needs --fresh.
    automap: bool = os.getenv("AUTOMAP", "1") in ("1", "true", "True")
    # Labels buffer: ground-truth on-screen enemy detection (ViZDoom labels). Does NOT change
    # the obs shape — used for telemetry and an optional engagement reward. No --fresh needed.
    use_labels: bool = os.getenv("USE_LABELS", "1") in ("1", "true", "True")
    # Game-vars in the policy: feed normalised HEALTH+AMMO into the network (DFP/Arnold). The
    # agent currently can't SEE its own health → keeps fighting until it dies at low HP. With
    # this it learns to retreat when weak. Makes the obs a Dict → MultiInputPolicy, needs --fresh.
    game_vars: bool = os.getenv("GAME_VARS", "1") in ("1", "true", "True")
    # Intrinsic curiosity (RND): spatial bonus that never saturates (unlike count-based
    # coverage_reward which dries up once the starting room is covered). When enabled,
    # adds a normalised prediction-error bonus for visiting unfamiliar positions.
    # Use when exploration is stuck < 20% and coverage_reward alone isn't enough.
    use_rnd: bool = os.getenv("USE_RND", "1") in ("1", "true", "True")
    rnd_scale: float = float(os.getenv("RND_SCALE", "0.3"))
    # Go-Explore "return, then explore": with prob goal_prob, an episode is handed a far,
    # rarely-seen frontier cell (from the persisted archive) as a goal; a dense potential
    # reward guides the agent BACK to it, then exploration takes over from that launch point.
    # 0 = off. Use when the agent is stuck near spawn and never reaches the far map / exit.
    goexplore_goal_prob: float = float(os.getenv("GOEXPLORE_GOAL_PROB", "0.4"))
    goexplore_goal_scale: float = float(os.getenv("GOEXPLORE_GOAL_SCALE", "0.01"))
    goexplore_reach_radius: float = float(os.getenv("GOEXPLORE_REACH_RADIUS", "96.0"))
    # Recurrent policy (LSTM): give the policy temporal memory across steps via
    # RecurrentPPO (sb3-contrib). Opt-in — it changes the saved brain's format, so an
    # LSTM brain and a feed-forward brain are NOT interchangeable (checkpoint is tagged).
    use_lstm: bool = os.getenv("USE_LSTM", "0") in ("1", "true", "True")

    # ---------- Outputs ----------
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    tensorboard_log: str = os.getenv("TENSORBOARD_LOG", "./tb")
    # Where the run's snapshots live until post-processing (which writes the notes).
    pending_dir: str = os.getenv("PENDING_DIR", "./.cache/pending_runs")

    # ---------- Documentation (local LLM via Ollama -> Obsidian) ----------
    vault_path: str = os.getenv("VAULT_PATH", "./vault")
    # Light default model (fast on M5 16GB; good enough for the notes).
    llm_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    write_every_steps: int = int(os.getenv("WRITE_EVERY_STEPS", "50000"))
    novelty_threshold: float = float(os.getenv("NOVELTY_THRESHOLD", "0.15"))
    # LLM cost knobs (the fact sheet is small, so a big context is wasteful):
    # smaller num_ctx = less memory; num_predict caps generation = faster & bounded;
    # keep_alive keeps the model warm across the batch (no reload between notes).
    llm_num_ctx: int = int(os.getenv("LLM_NUM_CTX", "4096"))
    llm_num_predict: int = int(os.getenv("LLM_NUM_PREDICT", "700"))
    llm_keep_alive: str = os.getenv("LLM_KEEP_ALIVE", "5m")
    # Max new concept notes to generate per checkpoint (each is an extra LLM call).
    max_new_concepts_per_ckpt: int = int(os.getenv("MAX_NEW_CONCEPTS_PER_CKPT", "2"))

    # ---------- Cognitive memory (persists ACROSS runs) ----------
    # Records episode-end events (death/success/timeout) during training (cheap,
    # async-safe) and, post-training, an LLM extracts reusable "lessons" from them.
    memory_enabled: bool = os.getenv("MEMORY_ENABLED", "1") not in ("0", "false", "False")
    memory_dir: str = os.getenv("MEMORY_DIR", "")  # default tied to vault in __post_init__
    min_events_for_lessons: int = int(os.getenv("MIN_EVENTS_FOR_LESSONS", "10"))
    # Phase 6: offline LLM proposes reward-weight tweaks (human approves; NEVER auto-applied).
    suggest_rewards: bool = os.getenv("SUGGEST_REWARDS", "1") not in ("0", "false", "False")

    def reward_weights(self) -> dict:
        """The tunable reward-shaping weights, passed into the envs."""
        return {
            "hit_reward": self.hit_reward,
            "kill_reward": self.kill_reward,
            "miss_penalty": self.miss_penalty,
            "damage_taken_penalty": self.damage_taken_penalty,
            "death_penalty": self.death_penalty,
            "move_reward": self.move_reward,
            "living_reward": self.living_reward,
            "frontier_reward": self.frontier_reward,
            "coverage_reward": self.coverage_reward,
            "coverage_cell": self.coverage_cell,
            "exit_reward": self.exit_reward,
            "exit_prox_scale": self.exit_prox_scale,
            "engagement_reward": self.engagement_reward,
            "combat_explore_split": float(self.combat_explore_split),
            "combat_explore_factor": self.combat_explore_factor,
            "auto_use": float(self.auto_use),
            "discovery_reward": self.discovery_reward,
            "weapon_variety_reward": self.weapon_variety_reward,
            "use_rnd":  float(self.use_rnd),
            "rnd_scale": self.rnd_scale,
            "goexplore_goal_prob": self.goexplore_goal_prob,
            "goexplore_goal_scale": self.goexplore_goal_scale,
            "goexplore_reach_radius": self.goexplore_reach_radius,
        }
    # Feedback loop: training re-reads 00-index/control.md every N steps and adapts
    # (stop_training, novelty_threshold, write_every_steps) without restarting.
    control_enabled: bool = os.getenv("CONTROL_ENABLED", "1") not in ("0", "false", "False")
    control_every_steps: int = int(os.getenv("CONTROL_EVERY_STEPS", "4096"))
    # When True, do NOT call the LLM or write notes (pure training, lighter).
    docs_enabled: bool = os.getenv("DOCS_ENABLED", "1") not in ("0", "false", "False")
    # When True, open the Doom window (forces 1 env, non-parallel).
    render: bool = os.getenv("RENDER", "0") in ("1", "true", "True")
    # run name (used in frontmatter and the index)
    run_name: str = os.getenv("RUN_NAME", "")

    # vault subfolders
    dir_index: str = "00-index"
    dir_checkpoints: str = "10-checkpoints"
    dir_concepts: str = "20-concepts"
    dir_runs: str = "30-runs"
    dir_maps: str = "40-maps"
    dir_compare: str = "50-compare"        # run-comparison notes
    dir_lessons: str = "60-lessons"        # cross-run lessons (cognitive memory)
    dir_attachments: str = "attachments"  # images (minimap) embedded in notes

    def __post_init__(self) -> None:
        if not self.run_name:
            from datetime import datetime
            self.run_name = "run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # Default WAD: the freedoom2.wad bundled with ViZDoom (resolved at runtime).
        if not self.wad_path:
            from doom.campaign import default_wad
            self.wad_path = default_wad()
        # The brain is TIED to the vault: same vault -> reuse; another vault -> start
        # from zero. (Unless CHECKPOINT_DIR is set explicitly.)
        if not os.getenv("CHECKPOINT_DIR"):
            self.checkpoint_dir = os.path.join(self.vault_path, ".checkpoints")
        # Memory persists across runs of the same vault (like the brain).
        if not self.memory_dir:
            self.memory_dir = os.path.join(self.vault_path, ".memory")
