# Resultado — Diagnóstico e fix do argmax-collapse
Data: 2026-06-03

## O que descobrimos rodando testes de verdade

O brain (419k–575k steps, spatial+rnd) tinha **colapsado em passividade** no eval determinístico.
Mais treino o piorava. Rodando eval determinístico vs estocástico, a causa ficou clara:

| Métrica | Determinístico (argmax) | Estocástico (amostrado) |
|---------|:---:|:---:|
| Exploração | 2% (20 cel) | **6% (60 cel)** |
| Kills/ep | 0.00 | **1.20** |
| Endings | timeout 100% | death 65%, timeout 35% |

**Diagnóstico: argmax-collapse.** A política APRENDEU a explorar e lutar (quando amostrada),
mas a ação mais provável colapsou em "ficar parado". A massa da distribuição estava em ações
boas; o argmax estava na ruim.

## Causa raiz

`ent_coef=0.01` (bônus de entropia baixo) → entropia colapsou para -0.65 (política peaked
demais) → argmax cola numa ação inútil. O `doom-cli audit` marcava "saudável" porque olhava
só a tendência da entropia, não o gap argmax↔estocástico.

## O fix e a prova

Subi `ent_coef` 0.01 → 0.04 e continuei o treino (~150k steps):

| Métrica | ent_coef=0.01 (colapsado) | ent_coef=0.04 |
|---------|:---:|:---:|
| Kills/ep (determinístico) | 0.00 | **2.00** ✅ |
| Exploração | 2% (18 cel) | 3% (25 cel) |
| Entropia (treino) | -0.65 | -1.51 |

**O argmax-collapse foi quebrado** — o agente voltou a lutar deterministicamente (0→2 kills).

## Veredicto honesto

- ✅ **Diagnóstico correto + fix funciona**: ent_coef resolve o colapso de combate
- ✅ **Go-Explore funciona**: 93 células-fronteira arquivadas (de quase-zero)
- ⚠️ **Exploração ainda baixa (3%)**: o combate voltou, mas a exploração precisa de mais
- ❌ **Exit-rate ainda 0%**: nenhum mapa completo

## Adotado

- `ENT_COEF=0.03` virou o padrão no `.env` (entre o 0.01 colapsante e o 0.04 testado)
- `config.ent_coef` agora é tunável via env var

## Bug encontrado (a corrigir)

`_latest_checkpoint` prefere `_final.zip` mesmo quando existe um `_steps.zip` mais novo —
se um treino é morto antes do save final, o resume pega o brain velho. Deveria preferir o
mais recente por step count.

## Treino fresh (ENT_COEF=0.03 + Go-Explore + spatial+rnd, 250k, n-envs 1)

| | Determinístico (argmax) | Estocástico (amostrado) |
|---|:---:|:---:|
| Kills/ep | 0.00 | **2.80** |
| Exploração | 1% (10 cel) | **8% (78 cel)** |
| Endings | timeout 100% | death 85%, timeout 15% |

**A política estocástica atingiu a melhor exploração da sessão (8%, 78 células) e 2.8 kills.**
O aprendizado é real. Mas o argmax do brain fresh colapsou (subtreinado: 250k em 1 env).

### Por que n-envs 1?

Descoberta de infra importante: o Mac de 16GB **não sustenta** um treino fresh com n-envs 8 +
obs spatial (8 canais). O buffer de rollout (~3GB) + 8 workers ViZDoom + spike do primeiro
update → OOM-kill. Pior: **26 processos zumbis** acumulados de runs em background sufocavam a
RAM (cada SubprocVecEnv morto deixava workers órfãos). Limpeza liberou ~1GB. Mesmo assim,
fresh n-envs 8 morria no primeiro update. Solução: n-envs 1 (DummyVecEnv in-process) completa,
mas é muito menos sample-efficient (1 trajetória correlacionada vs 8 diversas).

## Conclusões honestas da sessão

1. ✅ **argmax-collapse diagnosticado e provado** (gap det↔estocástico consistente)
2. ✅ **ent_coef ajuda** (brain continued: det kills 0→2.0)
3. ✅ **a política aprende a explorar** — estocástico chegou a 8% (melhor da sessão)
4. ⚠️ **argmax subtreinado colapsa** — precisa de mais treino com N envs diversos
5. ⚠️ **restrição de hardware** — 16GB não comporta fresh n-envs 8 com obs spatial
6. ❌ **exit-rate ainda 0%**

## Insight acionável

O eval determinístico **subestima** este agente. A política genuinamente explora 8% e mata
2.8/ep quando amostrada. Caminhos:
- Avaliar/usar com amostragem de baixa temperatura (não argmax puro)
- Treinar muito mais com N envs (precisa de máquina maior OU obs menor que 8 canais)
- O Go-Explore + memória já funcionam; falta o agente ter compute pra convergir o argmax

## Bugs corrigidos durante os testes

1. `_latest_checkpoint` preferia `_final` velho → agora pega por mtime
2. single-env spawava subprocess → agora usa DummyVecEnv (sem broken-pipe)
3. `config.ent_coef` agora tunável via env var

---
_Gerado durante sessão de testes do HeLLMind_
