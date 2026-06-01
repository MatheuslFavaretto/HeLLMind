"""Pós-processamento: lê os snapshots coletados no treino e gera as notas.

Roda DEPOIS do treino (o `rl.train` chama isto automaticamente no fim), então o
Ollama nunca trava o loop do PPO. Também dá pra rodar à mão:

    python -m writer.process_run                 # processa a run do .env (RUN_NAME)
    python -m writer.process_run --run NOME      # processa uma run específica
    python -m writer.process_run --model qwen2.5:7b   # usa um modelo melhor p/ as notas

Como o LLM agora roda em lote, vale usar um modelo MAIOR aqui (ex.: qwen2.5:7b):
o treino já acabou, então não há custo de velocidade — só notas melhores.
"""
import argparse
from typing import List, Optional

from config import Config
from writer.note_writer import NoteWriter
from writer.snapshot_log import (
    SnapshotLog,
    log_path_for,
    meta_path_for,
    read_meta,
)


def process_run(
    cfg: Config,
    button_names: List[str],
    log_path: Optional[str] = None,
) -> int:
    """Gera as notas de todos os snapshots da run. Retorna quantas notas escreveu."""
    log_path = log_path or log_path_for(cfg.pending_dir, cfg.run_name)
    snaps = SnapshotLog.read_all(log_path)
    if not snaps:
        print(f"[process_run] nenhum snapshot em {log_path} — nada a gerar.")
        return 0

    print(
        f"[process_run] {len(snaps)} snapshot(s) | modelo: {cfg.llm_model} | "
        f"vault: {cfg.vault_path}\n[process_run] gerando notas (pode levar um tempo)..."
    )
    writer = NoteWriter(cfg, button_names=button_names)
    previous = None
    written = 0
    for i, snap in enumerate(snaps, 1):
        try:
            stem = writer.write_checkpoint(snap, previous=previous)
            written += 1
            print(f"[process_run] {i}/{len(snaps)} -> {stem}")
        except Exception as e:  # uma nota ruim não derruba o resto
            print(
                f"[process_run] {i}/{len(snaps)} FALHOU "
                f"(step={snap.get('num_timesteps')}): {e}"
            )
        previous = snap

    # Curva de aprendizado da run inteira (embutida na nota da run).
    try:
        chart = writer.write_run_chart(snaps)
        if chart:
            print(f"[process_run] curva de aprendizado: attachments/{chart}")
    except Exception as e:
        print(f"[process_run] curva falhou (ignorando): {e}")

    # (A) Síntese narrativa da run inteira.
    try:
        story = writer.write_run_story(snaps)
        if story:
            print(f"[process_run] síntese da run: {story}")
    except Exception as e:
        print(f"[process_run] síntese falhou (ignorando): {e}")

    print(f"[process_run] concluído: {written}/{len(snaps)} notas em {cfg.vault_path}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Gera as notas do Obsidian a partir dos snapshots.")
    p.add_argument("--run", default=None, help="Nome da run (default: RUN_NAME do .env).")
    p.add_argument("--model", default=None, help="Modelo Ollama p/ as notas (override).")
    args = p.parse_args()

    cfg = Config()
    if args.run:
        cfg.run_name = args.run
    if args.model:
        cfg.llm_model = args.model

    meta = read_meta(meta_path_for(cfg.pending_dir, cfg.run_name)) or {}
    button_names = meta.get("button_names", [])
    if meta.get("scenario"):
        cfg.scenario = meta["scenario"]
    process_run(cfg, button_names)


if __name__ == "__main__":
    main()
