"""Entrypoint: treina PPO no Doom e (opcionalmente) documenta no Obsidian.

Uso:
    python -m rl.train                 # treino + notas no vault (Ollama)
    python -m rl.train --no-docs       # treino puro, sem LLM/notas (mais leve)
    python -m rl.train --render        # abre a janela do Doom (1 env, mais lento)
    python -m rl.train --render --no-docs --timesteps 100000

Flags sobrescrevem o .env.
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
from writer.snapshot_log import (
    SnapshotLog,
    log_path_for,
    meta_path_for,
    write_meta,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina PPO no Doom + notas no Obsidian.")
    p.add_argument(
        "--no-docs",
        action="store_true",
        help="Não chama o LLM nem escreve notas (treino puro, mais leve).",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Abre a janela do Doom (força 1 env, não-paralelo, mais lento).",
    )
    p.add_argument("--model", type=str, default=None, help="Modelo Ollama (override).")
    p.add_argument("--timesteps", type=int, default=None, help="Total de timesteps.")
    p.add_argument("--n-envs", type=int, default=None, help="Nº de ambientes paralelos.")
    p.add_argument(
        "--campaign",
        action="store_true",
        help="Modo campanha: joga mapas completos de um WAD, em ordem.",
    )
    p.add_argument(
        "--maps",
        type=str,
        default=None,
        help="Lista de mapas separados por vírgula (ex.: MAP01,MAP02 ou E1M1,E1M2).",
    )
    p.add_argument("--wad", type=str, default=None, help="Caminho do WAD (campanha).")
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Continua de um checkpoint salvo (treino em lotes). Sem valor = "
        "pega o último .zip automaticamente; ou passe o caminho de um .zip.",
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
    # Render exige um único ambiente com janela; paralelo não dá pra visualizar.
    if cfg.render and cfg.n_envs != 1:
        print("[render] forçando n_envs=1 para mostrar a janela do Doom.")
        cfg.n_envs = 1
    return cfg


def _probe_ollama(cfg: Config) -> bool:
    """Verifica o Ollama SEM derrubar o treino. Retorna True se estiver pronto.

    O projeto funciona perfeitamente sem o Ollama: os snapshots são coletados de
    qualquer jeito e, no fim, as notas saem em modo factual (sem narrativa do LLM).
    """
    try:
        from ollama import Client

        client = Client(host=cfg.ollama_host)
        models = [m.model for m in client.list().models]
    except Exception:
        print(
            f"[docs] Ollama indisponível em {cfg.ollama_host} — seguindo assim mesmo.\n"
            f"       As notas sairão em modo FACTUAL (sem narrativa). Para narrativa,\n"
            f"       suba o `ollama serve` e rode depois: python -m writer.process_run"
        )
        return False
    wanted = cfg.llm_model if ":" in cfg.llm_model else cfg.llm_model + ":latest"
    if not any((m or "") == wanted for m in models):
        print(
            f"[docs] Modelo '{cfg.llm_model}' não encontrado no Ollama (baixe com "
            f"`ollama pull {cfg.llm_model}`). Notas sairão em modo FACTUAL."
        )
        return False
    print(f"[docs] Ollama OK em {cfg.ollama_host} | modelo: {cfg.llm_model}")
    return True


def _resolve_resume(cfg: Config, arg: Optional[str], name_prefix: str) -> Optional[str]:
    """Resolve o caminho do checkpoint p/ continuar o treino (modo lotes)."""
    if arg is None:
        return None
    if arg != "auto":
        path = arg if arg.endswith(".zip") else arg + ".zip"
        return path if os.path.exists(path) else None
    # auto: prioriza o _final; senão o .zip mais recente do prefixo.
    final = os.path.join(cfg.checkpoint_dir, f"{name_prefix}_final.zip")
    if os.path.exists(final):
        return final
    candidates = sorted(
        glob.glob(os.path.join(cfg.checkpoint_dir, f"{name_prefix}*.zip")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def build_vec_env(cfg: Config):
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
            )
            for rank in range(cfg.n_envs)
        ]
    # Render precisa de janela no processo principal -> DummyVecEnv (sem subprocess).
    venv = DummyVecEnv(env_fns) if cfg.render else SubprocVecEnv(env_fns)
    venv = VecMonitor(venv)  # garante stats de episódio agregados
    venv = VecFrameStack(venv, n_stack=cfg.frame_stack)  # empilha frames (movimento)
    return venv


def main() -> None:
    args = parse_args()
    cfg = apply_args(Config(), args)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # O Ollama é OPCIONAL: se faltar, o treino segue e as notas saem em modo factual.
    if cfg.docs_enabled:
        _probe_ollama(cfg)
    else:
        print("[docs] desabilitado — treino puro, sem chamar o LLM.")
    if cfg.render:
        print("[render] janela do Doom habilitada (1 env).")

    # Descobre nomes de botões (rótulos da distribuição de ações) sem subir o treino.
    if cfg.campaign:
        meta = campaign_metadata(cfg.wad_path, cfg.maps[0])
        button_names = meta["button_names"]
        print(
            f"Campanha | WAD: {os.path.basename(cfg.wad_path)} | "
            f"mapas: {list(cfg.maps)} | {cfg.steps_per_map} steps/mapa | "
            f"ações: {meta['num_actions']} {button_names}"
        )
    else:
        meta = probe_env_metadata(cfg.scenario, cfg.frame_skip, cfg.resolution)
        button_names = meta["button_names"]
        print(f"Cenário: {cfg.scenario} | ações: {meta['num_actions']} {button_names}")

    venv = build_vec_env(cfg)
    name_prefix = f"ppo_{'campaign' if cfg.campaign else cfg.scenario}"

    # Treino em LOTES: continua o "cérebro" salvo em vez de começar do zero.
    resume_path = _resolve_resume(cfg, args.resume, name_prefix)
    if resume_path:
        print(f"[resume] continuando o treino de {resume_path}")
        model = PPO.load(resume_path, env=venv, tensorboard_log=cfg.tensorboard_log)
        reset_timesteps = False
    else:
        if args.resume is not None:
            print("[resume] nenhum checkpoint encontrado — começando do zero.")
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

    callbacks = []
    # No modo campanha, o currículo troca de mapa por timesteps.
    if cfg.campaign and len(cfg.maps) > 1:
        callbacks.append(
            MapCurriculumCallback(
                maps=list(cfg.maps),
                steps_per_map=cfg.steps_per_map,
                loop_maps=cfg.loop_maps,
            )
        )
    log_path = log_path_for(cfg.pending_dir, cfg.run_name)
    doc_cb = None
    if cfg.docs_enabled:
        tracker = StatsTracker(button_names=button_names)
        snap_log = SnapshotLog(log_path)
        # Sidecar com metadados p/ o pós-processamento (writer.process_run).
        write_meta(
            meta_path_for(cfg.pending_dir, cfg.run_name),
            {
                "run_name": cfg.run_name,
                "scenario": cfg.scenario,
                "campaign": cfg.campaign,
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

    # Loop de feedback: o Obsidian (00-index/control.md) controla o treino ao vivo.
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
        print("[info] tqdm/rich ausentes — seguindo sem barra de progresso.")

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            progress_bar=use_bar,
            reset_num_timesteps=reset_timesteps,
        )
    finally:
        model.save(os.path.join(cfg.checkpoint_dir, f"{name_prefix}_final"))
        venv.close()

    # PÓS-TREINO: agora (e só agora) chamamos o LLM, em lote. O loop do PPO já
    # terminou, então gerar as notas não trava mais nada (conserto da travadinha).
    if cfg.docs_enabled:
        print("\n[docs] treino concluído — gerando notas no Obsidian (em lote)...")
        from writer.process_run import process_run

        process_run(cfg, button_names, log_path)


if __name__ == "__main__":
    main()
