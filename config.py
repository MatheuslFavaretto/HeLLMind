"""Central project configuration. Reads from the environment (.env) with sane defaults."""
import os
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ---------- Doom environment ----------
    scenario: str = os.getenv("DOOM_SCENARIO", "defend_the_center")
    frame_skip: int = 4
    frame_stack: int = 4
    resolution: Tuple[int, int] = (84, 84)  # (width, height)

    # ---------- CAMPAIGN mode (full WAD maps, in order) ----------
    # Enable campaign mode (train full maps and advance to the next).
    campaign: bool = os.getenv("CAMPAIGN", "0") in ("1", "true", "True")
    # WAD with the maps. Default: bundled freedoom2.wad (free).
    # For the original Doom 1 maps, point to your doom.wad.
    wad_path: str = os.getenv("WAD_PATH", "")
    # Map list, in order. freedoom2 uses MAP01..; doom.wad uses E1M1..
    maps: Tuple[str, ...] = tuple(
        os.getenv("MAPS", "MAP01,MAP02,MAP03,MAP04,MAP05").split(",")
    )
    steps_per_map: int = int(os.getenv("STEPS_PER_MAP", "200000"))
    loop_maps: bool = os.getenv("LOOP_MAPS", "0") in ("1", "true", "True")
    kills_to_clear: int = int(os.getenv("KILLS_TO_CLEAR", "5"))
    episode_timeout: int = int(os.getenv("EPISODE_TIMEOUT", "2100"))  # ticks

    # ---------- PPO training ----------
    total_timesteps: int = int(os.getenv("TOTAL_TIMESTEPS", "2000000"))
    n_envs: int = int(os.getenv("N_ENVS", "8"))
    n_steps: int = 1024          # rollout per env before each update
    batch_size: int = 2048
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    learning_rate: float = 2.5e-4
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
    frontier_reward: float = float(os.getenv("FRONTIER_REWARD", "0.0"))
    # ---------- Exploration & completion (autonomy goal: explore the whole map) ----------
    # Bonus the first time the agent steps on a NEW grid cell (count-based exploration).
    # Drives covering the map instead of pacing the same corridor.
    coverage_reward: float = float(os.getenv("COVERAGE_REWARD", "0.5"))
    # Grid cell size (map units) for both the coverage reward and the spatial memory.
    coverage_cell: float = float(os.getenv("COVERAGE_CELL", "96.0"))
    # Big bonus for actually reaching the level EXIT (episode ends, not dead, pre-timeout).
    exit_reward: float = float(os.getenv("EXIT_REWARD", "200.0"))
    # Count-based weapon variety: bonus the FIRST time the agent wields a NEW weapon slot
    # in an episode (campaign only). The campaign has SELECT_NEXT_WEAPON but, without this,
    # no reason to ever switch — so it never used the weapons it picks up.
    weapon_variety_reward: float = float(os.getenv("WEAPON_VARIETY_REWARD", "0.5"))
    # Closed loop (bestiary -> reward): scale the kill bonus by what the agent LEARNED about
    # each monster — killing a deadlier type (higher death-rate-when-present) pays more. Uses
    # the persisted bestiary; needs one prior run to have data. Opt-in (changes the reward).
    bestiary_reward: bool = os.getenv("BESTIARY_REWARD", "0") in ("1", "true", "True")
    # Spatial memory: feed the agent a 2nd obs channel of where it has already been
    # (so it can ACT on its own memory, not just be rewarded for it).
    spatial_memory: bool = os.getenv("SPATIAL_MEMORY", "0") in ("1", "true", "True")
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
            "weapon_variety_reward": self.weapon_variety_reward,
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
