"""Mostra o ESTADO do treino: o que foi aprendido (checkpoints) e como vai indo.

Esclarece uma dúvida comum: o `.zip` salvo em ./checkpoints é o "cérebro" (os pesos
da política PPO). O TAMANHO dele é ~constante — depende da arquitetura da rede, não
de "quanto o agente aprendeu". Quem diz se o treino está eficiente é a CURVA de
recompensa/precisão (veja a coluna de métricas abaixo, ou a curva na nota da run).

Uso:
    python -m rl.status                 # checkpoints + métricas da run do .env
    python -m rl.status --run NOME      # métricas de uma run específica
"""
import argparse
import glob
import os
from datetime import datetime

from config import Config
from writer.snapshot_log import SnapshotLog, log_path_for


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def show_checkpoints(cfg: Config) -> None:
    zips = sorted(
        glob.glob(os.path.join(cfg.checkpoint_dir, "*.zip")), key=os.path.getmtime
    )
    print(f"\n== Cérebros salvos em {cfg.checkpoint_dir} ==")
    if not zips:
        print("  (nenhum ainda — rode `python -m rl.train`)")
        return
    for z in zips:
        st = os.stat(z)
        when = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {os.path.basename(z):<40} {_human_size(st.st_size):>10}  {when}")
    print(
        "  Obs: o tamanho é ~constante (arquitetura da rede). "
        "Eficiência = curva de recompensa, não tamanho."
    )


def show_metrics(cfg: Config) -> None:
    snaps = SnapshotLog.read_all(log_path_for(cfg.pending_dir, cfg.run_name))
    print(f"\n== Progresso da run '{cfg.run_name}' ({len(snaps)} checkpoints) ==")
    if not snaps:
        print("  (sem snapshots — treine com a documentação ligada)")
        return
    first, last = snaps[0], snaps[-1]

    def line(s, tag):
        print(
            f"  [{tag}] {int(s.get('num_timesteps', 0)):>9,} steps | "
            f"recompensa {s.get('mean_reward', 0):6.2f} | "
            f"precisão {s.get('shooting_accuracy', 0):4.0%} | "
            f"kills/ep {s.get('kills_per_episode', 0):4.1f} | "
            f"sucesso {s.get('success_rate', 0):4.0%}"
        )

    line(first, "início")
    line(last, " atual ")
    d_reward = last.get("mean_reward", 0) - first.get("mean_reward", 0)
    d_acc = last.get("shooting_accuracy", 0) - first.get("shooting_accuracy", 0)
    seta = "📈" if d_reward >= 0 else "📉"
    print(f"  {seta} variação: recompensa {d_reward:+.2f} | precisão {d_acc:+.0%}")


def main() -> None:
    p = argparse.ArgumentParser(description="Estado do treino: checkpoints + métricas.")
    p.add_argument("--run", default=None, help="Run específica (default: RUN_NAME do .env).")
    args = p.parse_args()
    cfg = Config()
    if args.run:
        cfg.run_name = args.run
    show_checkpoints(cfg)
    show_metrics(cfg)


if __name__ == "__main__":
    main()
