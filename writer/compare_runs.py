"""Compara duas (ou mais) runs e escreve uma nota com tabela + gráficos + veredito.

Responde, por exemplo: "o reward shaping de pontaria ajudou?". Lê os snapshots já
coletados de cada run (mesmo JSONL das notas), resume as métricas, sobrepõe os
gráficos e — se o Ollama estiver disponível — pede um veredito ao LLM. Tudo
pós-treino, sem tocar no loop do PPO.

Uso:
    python -m writer.compare_runs --runs run-A run-B
    python -m writer.compare_runs --runs run-A run-B --labels "com shaping" "sem shaping"
"""
import argparse
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import Config
from writer.charts import render_run_comparison
from writer.note_writer import _yaml_frontmatter
from writer.snapshot_log import SnapshotLog, log_path_for

# Métricas resumidas (chave -> rótulo legível, é porcentagem?).
METRICS = [
    ("mean_reward", "Recompensa média/ep", False),
    ("shooting_accuracy", "Precisão de tiro", True),
    ("kills_per_episode", "Kills/episódio", False),
    ("success_rate", "Taxa de sucesso", True),
    ("distance_per_episode", "Distância/episódio", False),
]


def summarize(snaps: List[dict]) -> Dict:
    """Resumo (final/melhor/média) por métrica + metadados da run."""
    out: Dict = {"checkpoints": len(snaps),
                 "timesteps": int(snaps[-1].get("num_timesteps", 0)) if snaps else 0}
    for key, _label, _pct in METRICS:
        xs = [float(s.get(key, 0.0)) for s in snaps]
        out[key] = {
            "final": xs[-1] if xs else 0.0,
            "best": max(xs) if xs else 0.0,
            "mean": (sum(xs) / len(xs)) if xs else 0.0,
        }
    return out


def _fmt(v: float, pct: bool) -> str:
    return f"{v:.0%}" if pct else f"{v:,.2f}"


def _winner_table(labels: List[str], summaries: Dict[str, Dict]) -> str:
    """Tabela markdown comparando o valor FINAL de cada métrica entre as runs."""
    header = "| Métrica | " + " | ".join(labels) + " | Melhor |"
    sep = "|" + "---|" * (len(labels) + 2)
    rows = [header, sep]
    for key, label, pct in METRICS:
        finals = [summaries[l][key]["final"] for l in labels]
        best_idx = max(range(len(finals)), key=lambda i: finals[i])
        cells = " | ".join(_fmt(v, pct) for v in finals)
        rows.append(f"| {label} | {cells} | **{labels[best_idx]}** |")
    return "\n".join(rows)


def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "-", text).strip("-")


def compare_runs(
    cfg: Config, run_names: List[str], labels: Optional[List[str]] = None
) -> Optional[str]:
    """Gera a nota de comparação. Retorna o stem do arquivo, ou None se faltarem dados."""
    runs: Dict[str, List[dict]] = {}
    for rn in run_names:
        snaps = SnapshotLog.read_all(log_path_for(cfg.pending_dir, rn))
        if snaps:
            runs[rn] = snaps
        else:
            print(f"[compare] run '{rn}' sem snapshots em {cfg.pending_dir} — pulando.")
    if len(runs) < 2:
        print("[compare] preciso de pelo menos 2 runs com dados. Abortando.")
        return None

    labels = labels or list(runs.keys())
    labels = labels[: len(runs)]
    by_label = {labels[i]: snaps for i, (_rn, snaps) in enumerate(runs.items())}
    summaries = {label: summarize(snaps) for label, snaps in by_label.items()}

    dir_compare = os.path.join(cfg.vault_path, cfg.dir_compare)
    dir_attach = os.path.join(cfg.vault_path, cfg.dir_attachments)
    os.makedirs(dir_compare, exist_ok=True)

    stem = "Compare - " + _slug(" vs ".join(labels))[:80]

    # Gráficos sobrepostos (recompensa e precisão).
    charts_md = ""
    for key, title in (("mean_reward", "Recompensa média/ep"),
                       ("shooting_accuracy", "Precisão de tiro")):
        img = f"{stem} - {key}.png"
        if render_run_comparison(by_label, key, os.path.join(dir_attach, img), title=title):
            charts_md += f"![[{img}]]\n\n"

    # Veredito: heurístico (sempre) + LLM (se disponível).
    finals_reward = {l: summaries[l]["mean_reward"]["final"] for l in labels}
    heur_winner = max(finals_reward, key=finals_reward.get)
    verdict_md = f"**Veredito (heurístico, por recompensa final):** {heur_winner}\n\n"
    try:
        from writer.llm_client import LLMWriter

        llm = LLMWriter(model=cfg.llm_model, host=cfg.ollama_host)
        v = llm.generate_comparison(labels, summaries)
        verdict_md += (
            f"**Veredito do LLM:** {v.winner}\n\n"
            f"{v.summary}\n\n_{v.reasoning}_\n\n"
        )
    except Exception as e:
        verdict_md += f"_(LLM indisponível — só o veredito heurístico. {e})_\n\n"

    fm = _yaml_frontmatter({
        "type": "comparison",
        "runs": list(run_names),
        "labels": list(labels),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tags": ["comparison", "doom-rl"],
    })
    run_links = ", ".join(f"[[{rn}]]" for rn in run_names)
    body = (
        f"# Comparação: {' vs '.join(labels)}\n\n"
        f"Runs: {run_links}\n\n"
        f"## Veredito\n\n{verdict_md}"
        f"## Métricas (valor final)\n\n{_winner_table(labels, summaries)}\n\n"
        f"## Evolução\n\n{charts_md}"
    )
    path = os.path.join(dir_compare, f"{stem}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(fm + "\n\n" + body)
    print(f"[compare] nota escrita: {os.path.join(cfg.dir_compare, stem)}.md")
    return stem


def main() -> None:
    p = argparse.ArgumentParser(description="Compara runs e escreve uma nota no vault.")
    p.add_argument("--runs", nargs="+", required=True, help="Nomes das runs (>=2).")
    p.add_argument("--labels", nargs="+", default=None, help="Rótulos legíveis (opcional).")
    p.add_argument("--model", default=None, help="Modelo Ollama p/ o veredito (override).")
    args = p.parse_args()

    cfg = Config()
    if args.model:
        cfg.llm_model = args.model
    compare_runs(cfg, args.runs, args.labels)


if __name__ == "__main__":
    main()
