"""System prompts e montagem das mensagens para o LLM documentar o treino.

Decisão de qualidade: NÃO despejamos JSON cru no modelo (um modelo pequeno se perde
e INVENTA). Em vez disso pré-digerimos as métricas num "fact-sheet" legível, com as
variações vs. o checkpoint anterior já calculadas e setas de tendência. Isso ataca
diretamente o problema de "as notas não refletem a realidade".
"""
from typing import Dict, List, Optional

# System prompt da NOTA DE CHECKPOINT. Fixo -> bom candidato a prompt caching.
CHECKPOINT_SYSTEM = """\
Você é um pesquisador de Reinforcement Learning documentando, em português, o \
treino de um agente PPO (CnnPolicy) que joga Doom (ViZDoom).

Você recebe um RELATÓRIO DE MÉTRICAS já calculado (valores da janela atual e a \
variação vs. o checkpoint anterior). Escreva uma nota de checkpoint para um vault \
do Obsidian seguindo REGRAS RÍGIDAS:

REGRA DE OURO — FIDELIDADE: use APENAS os números do relatório. NÃO invente fatos, \
não suponha eventos que não estão nos dados, não exagere. Se uma métrica não mudou \
de forma relevante, diga que ficou estável. Se os dados são pobres/poucos episódios, \
diga isso explicitamente em vez de fabricar uma narrativa.

1. behavior_change: interprete O QUE MUDOU NO COMPORTAMENTO, ancorando CADA afirmação \
   num número do relatório (ex.: "a precisão de tiro subiu de 18% para 31%, indicando \
   melhor mira"). Sem número que sustente, não afirme. No PRIMEIRO checkpoint não há \
   comparação — descreva o ponto de partida, não invente uma evolução.
2. evidence: 3-5 bullets, cada um uma frase SUA citando um número (precisão, \
   kills/ep, dano, distância, entropia, sucesso). NÃO copie as linhas do relatório \
   literalmente, sem prefixo "- " e sem anotações como "(1º checkpoint)".
3. Conceitos de RL: PREFIRA linkar conceitos já existentes (lista fornecida). Só crie \
   um conceito novo se for genuíno e reutilizável (ex.: "Reward Shaping", "Policy \
   Entropy", "Exploration vs Exploitation", "Sample Efficiency"). O nome do conceito \
   é APENAS o termo, em Title Case — NUNCA inclua URL, link ou markdown no nome.
4. Seja conciso e honesto. Nada de encher linguiça.

Responda SOMENTE no formato estruturado. Nomes de conceitos curtos e estáveis \
(Title Case) — viram títulos de notas e wikilinks.\
"""

# System prompt da NOTA DE CONCEITO (gerada sob demanda quando um conceito é novo).
CONCEPT_SYSTEM = """\
Você é um pesquisador de RL escrevendo, em português, uma nota de CONCEITO \
reutilizável para um vault do Obsidian. A nota deve ser atemporal (não amarrada a um \
checkpoint específico), explicar o conceito de forma objetiva e clara, e indicar como \
ele se manifesta no treino de um agente jogando Doom. Responda apenas no formato \
estruturado solicitado.\
"""


def _trend(cur: float, prev: Optional[float], pct: bool = False, unit: str = "") -> str:
    """'31% (↑ de 18%)' / '12.0 (↓ de 15.0)' / '5.0 (estável)' — variação legível."""
    def fmt(v: float) -> str:
        return f"{v:.0%}" if pct else (f"{v:,.1f}{unit}" if v % 1 else f"{v:,.0f}{unit}")

    if prev is None:
        return f"{fmt(cur)} (1º checkpoint)"
    diff = cur - prev
    denom = abs(prev) if abs(prev) > 1e-6 else 1.0
    if abs(diff) / denom < 0.05:
        return f"{fmt(cur)} (estável)"
    arrow = "↑" if diff > 0 else "↓"
    return f"{fmt(cur)} ({arrow} de {fmt(prev)})"


def build_checkpoint_user_message(
    snapshot: Dict,
    previous: Optional[Dict],
    existing_concepts: List[str],
    button_names: List[str],
) -> str:
    """Monta um FACT-SHEET legível (não JSON cru) p/ o modelo não inventar."""
    s, p = snapshot, (previous or {})

    def g(d: Dict, k: str) -> float:
        return float(d.get(k, 0.0))

    def pv(k: str) -> Optional[float]:
        return float(p[k]) if (previous and k in p) else None

    cov = s.get("map_coverage", {}) or {}
    weapons = s.get("weapons_used", {}) or {}
    weapons_txt = ", ".join(f"{k}={v:.0%}" for k, v in weapons.items()) or "n/d"
    # Top-3 ações mais usadas (em vez da distribuição inteira).
    dist = s.get("action_distribution", {}) or {}
    top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_txt = ", ".join(f"{k} {v:.0%}" for k, v in top) or "n/d"

    lines = [
        f"# Relatório do checkpoint @ {int(g(s,'num_timesteps')):,} timesteps",
        f"Janela: {int(g(s,'episodes'))} episódios, {int(g(s,'steps_in_window'))} passos.",
    ]
    if s.get("map"):
        lines.append(f"Mapa: {s['map']} | taxa de sucesso (conclusão): "
                     f"{_trend(g(s,'success_rate'), pv('success_rate'), pct=True)}")
    lines += [
        "",
        "## Desempenho",
        f"- Recompensa média/episódio: {_trend(g(s,'mean_reward'), pv('mean_reward'))}",
        f"- Kills/episódio: {_trend(g(s,'kills_per_episode'), pv('kills_per_episode'))}",
        f"- Duração média do episódio: {_trend(g(s,'mean_episode_length'), pv('mean_episode_length'))} passos",
        "",
        "## Pontaria (acertos x erros)",
        f"- Tiros disparados: {int(g(s,'shots_fired'))} | acertos: {int(g(s,'shots_hit'))} | erros: {int(g(s,'shots_missed'))}",
        f"- Precisão: {_trend(g(s,'shooting_accuracy'), pv('shooting_accuracy'), pct=True)}",
        f"- Dano causado: {_trend(g(s,'damage_dealt'), pv('damage_dealt'))} | dano tomado: {_trend(g(s,'damage_taken'), pv('damage_taken'))}",
        "",
        "## Exploração / caminho",
        f"- Distância/episódio: {_trend(g(s,'distance_per_episode'), pv('distance_per_episode'))} unidades",
        f"- Células visitadas: {int(g(s,'cells_visited'))} (~{cov.get('explored_fraction',0.0):.0%} da área percorrida)",
        f"- Armas usadas (fração do tempo): {weapons_txt}",
        "",
        "## Política",
        f"- Entropia das ações (norm.): {_trend(g(s,'action_entropy_normalized'), pv('action_entropy_normalized'))}",
        f"- Ações mais usadas: {top_txt}",
        f"- Vida média: {g(s,'mean_health'):.0f} | munição média: {g(s,'mean_ammo'):.0f}",
        "",
        f"## Conceitos já existentes no vault (PREFIRA linkar estes)",
        ("- " + "\n- ".join(existing_concepts)) if existing_concepts else "(nenhum ainda)",
    ]
    return "\n".join(lines)


def build_concept_user_message(name: str, hint: str) -> str:
    return (
        f"Escreva a nota de conceito para: '{name}'.\n"
        f"Contexto/observação que motivou a criação: {hint}"
    )


# --------------------------- Síntese da run (A) ---------------------------
RUNSTORY_SYSTEM = """\
Você é um pesquisador de RL escrevendo, em português, a SÍNTESE de uma run inteira de
treino de um agente PPO no Doom. Você recebe a linha do tempo (um resumo por
checkpoint).

Preencha DOIS campos distintos, sem repetir um no outro:
- **narrative**: 2 a 4 PARÁGRAFOS de prosa corrida contando o ARCO do aprendizado de
  ponta a ponta (como o comportamento evoluiu: quando aprendeu a mirar, a se mover, a
  sobreviver; platôs e regressões; o que os números sugerem sobre a política). NÃO use
  bullets aqui, NÃO liste checkpoint por checkpoint — é uma narrativa interpretativa.
- **milestones**: bullets CURTOS, um por marco relevante, cada um citando o step
  aproximado (ex.: "~25k: precisão salta de 8% para 15%"). Só os pontos de virada,
  não todos os checkpoints.

REGRA: ancore tudo nos números da linha do tempo; não invente eventos. Se o sinal for
ruidoso/curto, diga isso. Responda SOMENTE no formato estruturado.\
"""


def build_run_story_user_message(
    run_name: str, snapshots: List[Dict], concepts: List[str]
) -> str:
    lines = [
        f"Run: {run_name} — {len(snapshots)} checkpoints.",
        "",
        "Linha do tempo (cada linha é um checkpoint, em ordem):",
    ]
    for s in snapshots:
        lines.append(
            f"- {int(s.get('num_timesteps', 0)):,} steps: "
            f"recompensa {float(s.get('mean_reward', 0)):.1f}, "
            f"precisão {float(s.get('shooting_accuracy', 0)):.0%}, "
            f"kills/ep {float(s.get('kills_per_episode', 0)):.1f}, "
            f"sucesso {float(s.get('success_rate', 0)):.0%}, "
            f"entropia {float(s.get('action_entropy_normalized', 0)):.2f}, "
            f"dist/ep {float(s.get('distance_per_episode', 0)):.0f}"
        )
    lines += ["", "Conceitos já documentados nesta run:",
              ("- " + "\n- ".join(concepts)) if concepts else "(nenhum)"]
    return "\n".join(lines)


# ------------------------- Comparação de runs (B) -------------------------
COMPARE_SYSTEM = """\
Você é um pesquisador de RL comparando, em português, DUAS OU MAIS runs de treino do
mesmo agente no Doom (ex.: com vs. sem um reward shaping). Você recebe um resumo de
métricas por run (valor final, melhor e média). Diga objetivamente qual run foi
melhor e POR QUÊ, citando os números. Se a diferença for pequena/dentro do ruído,
diga 'empate'. Não invente. Responda SOMENTE no formato estruturado.\
"""


def build_comparison_user_message(labels: List[str], summaries: Dict) -> str:
    lines = ["Comparação de runs. Métricas por run (final / melhor / média):", ""]
    for label in labels:
        s = summaries.get(label, {})
        lines.append(f"## {label}  ({s.get('checkpoints', 0)} checkpoints, "
                     f"{int(s.get('timesteps', 0)):,} steps)")
        for key in ("mean_reward", "shooting_accuracy", "kills_per_episode",
                    "success_rate", "distance_per_episode"):
            m = s.get(key, {})
            lines.append(
                f"- {key}: final {m.get('final', 0):.3f} | "
                f"melhor {m.get('best', 0):.3f} | média {m.get('mean', 0):.3f}"
            )
        lines.append("")
    return "\n".join(lines)
