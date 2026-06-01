"""Configuração central do projeto. Lê do ambiente (.env) com defaults sensatos."""
import os
from dataclasses import dataclass, field
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ---------- Ambiente Doom ----------
    scenario: str = os.getenv("DOOM_SCENARIO", "defend_the_center")
    frame_skip: int = 4
    frame_stack: int = 4
    resolution: Tuple[int, int] = (84, 84)  # (largura, altura)

    # ---------- Modo CAMPANHA (mapas completos de um WAD, em ordem) ----------
    # Liga o modo campanha (treina mapas inteiros e avança para o próximo).
    campaign: bool = os.getenv("CAMPAIGN", "0") in ("1", "true", "True")
    # WAD com os mapas. Default: freedoom2.wad embutido (gratuito).
    # Para os mapas do Doom 1 originais, aponte para o seu doom.wad.
    wad_path: str = os.getenv("WAD_PATH", "")
    # Lista de mapas, em ordem. freedoom2 usa MAP01..; doom.wad usa E1M1..
    maps: Tuple[str, ...] = tuple(
        os.getenv("MAPS", "MAP01,MAP02,MAP03,MAP04,MAP05").split(",")
    )
    steps_per_map: int = int(os.getenv("STEPS_PER_MAP", "200000"))
    loop_maps: bool = os.getenv("LOOP_MAPS", "0") in ("1", "true", "True")
    kills_to_clear: int = int(os.getenv("KILLS_TO_CLEAR", "5"))
    episode_timeout: int = int(os.getenv("EPISODE_TIMEOUT", "2100"))  # ticks

    # ---------- Treino PPO ----------
    total_timesteps: int = int(os.getenv("TOTAL_TIMESTEPS", "2000000"))
    n_envs: int = int(os.getenv("N_ENVS", "8"))
    n_steps: int = 1024          # rollout por env antes de cada update
    batch_size: int = 2048
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    learning_rate: float = 2.5e-4
    clip_range: float = 0.2
    seed: int = 42

    # ---------- Saídas ----------
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    tensorboard_log: str = os.getenv("TENSORBOARD_LOG", "./tb")
    # Onde os snapshots da run ficam até o pós-processamento (gera as notas).
    pending_dir: str = os.getenv("PENDING_DIR", "./.cache/pending_runs")

    # ---------- Documentação (LLM local via Ollama -> Obsidian) ----------
    vault_path: str = os.getenv("VAULT_PATH", "./vault")
    # Modelo leve por padrão (rápido no M5 16GB; bom o bastante p/ as notas).
    llm_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    write_every_steps: int = int(os.getenv("WRITE_EVERY_STEPS", "50000"))
    novelty_threshold: float = float(os.getenv("NOVELTY_THRESHOLD", "0.15"))
    # Loop de feedback: o treino relê 00-index/control.md a cada N steps e se
    # adapta (stop_training, novelty_threshold, write_every_steps) sem reiniciar.
    control_enabled: bool = os.getenv("CONTROL_ENABLED", "1") not in ("0", "false", "False")
    control_every_steps: int = int(os.getenv("CONTROL_EVERY_STEPS", "4096"))
    # Quando True, NÃO chama o LLM nem escreve notas (treino puro, mais leve).
    docs_enabled: bool = os.getenv("DOCS_ENABLED", "1") not in ("0", "false", "False")
    # Quando True, abre a janela do Doom (força 1 env, não-paralelo).
    render: bool = os.getenv("RENDER", "0") in ("1", "true", "True")
    # nome da run (usado em frontmatter e no índice)
    run_name: str = os.getenv("RUN_NAME", "")

    # subpastas do vault
    dir_index: str = "00-index"
    dir_checkpoints: str = "10-checkpoints"
    dir_concepts: str = "20-concepts"
    dir_runs: str = "30-runs"
    dir_maps: str = "40-maps"
    dir_compare: str = "50-compare"        # notas de comparação entre runs
    dir_attachments: str = "attachments"  # imagens (minimapa) embutidas nas notas

    def __post_init__(self) -> None:
        if not self.run_name:
            from datetime import datetime
            self.run_name = "run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # WAD padrão: freedoom2.wad embutido no ViZDoom (resolvido em runtime).
        if not self.wad_path:
            from doom.campaign import default_wad
            self.wad_path = default_wad()
