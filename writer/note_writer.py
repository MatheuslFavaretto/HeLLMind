"""Orquestra a escrita das notas .md no vault (checkpoints + conceitos).

O Obsidian lê arquivos .md diretamente — não precisa de plugin nem API. Os
wikilinks [[Nome]] referenciam o STEM do arquivo (sem .md).
"""
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

from config import Config
from writer.analysis import (
    FORGETTING_CONCEPT,
    FORGETTING_DESCRIPTION,
    detect_regressions,
)
from writer.concept_registry import ConceptRegistry, clean_concept_name
from writer.llm_client import CheckpointNote, ConceptNote, LLMWriter
from writer.minimap import render_minimap


def _one_line(text: str, max_len: int) -> str:
    """1ª linha, sem markdown de cabeçalho/citação, truncada — p/ título/headline."""
    s = str(text or "").strip()
    s = s.split("\n", 1)[0].lstrip("#> ").strip()
    return s[:max_len].strip()


def _slug_concept(name: str) -> str:
    """Nome do arquivo de uma nota de conceito (= alvo do wikilink, sem .md)."""
    safe = re.sub(r"[^\w\s\-]", "", clean_concept_name(name)).strip()
    return f"Concept - {safe}"


def _yaml_frontmatter(fields: dict) -> str:
    lines = ["---"]
    for k, v in fields.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


class NoteWriter:
    def __init__(self, cfg: Config, button_names: List[str]) -> None:
        self.cfg = cfg
        self.button_names = button_names
        self.llm = LLMWriter(model=cfg.llm_model, host=cfg.ollama_host)

        self.dir_ckpt = os.path.join(cfg.vault_path, cfg.dir_checkpoints)
        self.dir_concept = os.path.join(cfg.vault_path, cfg.dir_concepts)
        self.dir_runs = os.path.join(cfg.vault_path, cfg.dir_runs)
        self.dir_maps = os.path.join(cfg.vault_path, cfg.dir_maps)
        self.dir_attach = os.path.join(cfg.vault_path, cfg.dir_attachments)
        for d in (self.dir_ckpt, self.dir_concept, self.dir_runs,
                  self.dir_maps, self.dir_attach):
            os.makedirs(d, exist_ok=True)

        self.registry = ConceptRegistry(
            os.path.join(cfg.vault_path, ".concept_registry.json")
        )
        self._ckpt_index = 0
        self._last_ckpt_stem: Optional[str] = None
        self._seen_maps: set = set()
        self._ensure_run_note()

    # ------------------------------------------------------------------
    def _ensure_run_note(self) -> None:
        path = os.path.join(self.dir_runs, f"{self.cfg.run_name}.md")
        if os.path.exists(path):
            return
        fm = _yaml_frontmatter(
            {
                "type": "run",
                "scenario": self.cfg.scenario,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tags": ["run", "doom-rl"],
            }
        )
        body = (
            f"# Run: {self.cfg.run_name}\n\n"
            f"Treino PPO no cenário **{self.cfg.scenario}**.\n\n"
            f"## Checkpoints\n\n"
            f"_As notas de checkpoint desta run aparecem aqui conforme o treino avança._\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)

    def _append_to_run(self, ckpt_stem: str, headline: str) -> None:
        path = os.path.join(self.dir_runs, f"{self.cfg.run_name}.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"- [[{ckpt_stem}]] — {headline}\n")

    def write_run_chart(self, snapshots: List[dict]) -> Optional[str]:
        """Renderiza a curva de aprendizado da run e embute na nota da run."""
        from writer.charts import render_learning_curve

        img_name = f"{self.cfg.run_name}-curva.png"
        out = os.path.join(self.dir_attach, img_name)
        if not render_learning_curve(snapshots, out):
            return None
        path = os.path.join(self.dir_runs, f"{self.cfg.run_name}.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## Curva de aprendizado\n\n![[{img_name}]]\n")
        return img_name

    def write_run_story(self, snapshots: List[dict]) -> Optional[str]:
        """(A) Síntese narrativa da run inteira; linka da nota da run."""
        if not snapshots:
            return None

        # LLM conta o arco; se falhar, caímos num resumo factual mínimo.
        try:
            story = self.llm.generate_run_story(
                self.cfg.run_name, snapshots, self.registry.names()
            )
            title, narrative = story.title, story.narrative
            milestones, key_concepts = story.milestones, story.key_concepts
        except Exception:
            first, last = snapshots[0], snapshots[-1]
            title = f"Síntese: {self.cfg.run_name}"
            narrative = (
                f"Run com {len(snapshots)} checkpoints, de "
                f"{int(first.get('num_timesteps', 0)):,} a "
                f"{int(last.get('num_timesteps', 0)):,} steps. "
                f"Precisão de tiro foi de {first.get('shooting_accuracy', 0):.0%} "
                f"para {last.get('shooting_accuracy', 0):.0%}; recompensa média de "
                f"{first.get('mean_reward', 0):.1f} para {last.get('mean_reward', 0):.1f}."
            )
            milestones, key_concepts = [], []

        # Linka conceitos existentes pelo nome canônico (sem criar novos aqui).
        concept_links = [
            _slug_concept(self.registry.canonical(c))
            for c in key_concepts
            if self.registry.exists(c)
        ]
        concepts_md = "\n".join(f"- [[{c}]]" for c in dict.fromkeys(concept_links)) or "_(nenhum)_"
        milestones_md = "\n".join(f"- {m}" for m in milestones) or "_(n/d)_"

        stem = f"{self.cfg.run_name} - Síntese"
        fm = _yaml_frontmatter(
            {
                "type": "synthesis",
                "run": self.cfg.run_name,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "checkpoints": len(snapshots),
                "tags": ["synthesis", "doom-rl"],
            }
        )
        body = (
            f"# {title}\n\n"
            f"**Run:** [[{self.cfg.run_name}]]\n\n"
            f"{narrative}\n\n"
            f"## Marcos do treino\n\n{milestones_md}\n\n"
            f"## Conceitos centrais\n\n{concepts_md}\n"
        )
        with open(os.path.join(self.dir_runs, f"{stem}.md"), "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)

        run_path = os.path.join(self.dir_runs, f"{self.cfg.run_name}.md")
        with open(run_path, "a", encoding="utf-8") as f:
            f.write(f"\n## Síntese\n\n- [[{stem}]]\n")
        return stem

    # ------------------------------------------------------------------
    def _map_stem(self, doom_map: str) -> str:
        """Stem (alvo de wikilink) da nota de um mapa, ex.: 'Map - MAP01'."""
        return f"Map - {doom_map}"

    def _ensure_map_note(self, doom_map: str) -> str:
        """Cria a nota do mapa (uma por mapa) se ainda não existir; retorna o stem."""
        stem = self._map_stem(doom_map)
        path = os.path.join(self.dir_maps, f"{stem}.md")
        if doom_map in self._seen_maps or os.path.exists(path):
            self._seen_maps.add(doom_map)
            return stem
        fm = _yaml_frontmatter(
            {
                "type": "map",
                "map": doom_map,
                "run": self.cfg.run_name,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tags": ["map", "doom-rl"],
            }
        )
        body = (
            f"# Mapa: {doom_map}\n\n"
            f"Progresso do agente no mapa **{doom_map}** (run [[{self.cfg.run_name}]]).\n\n"
            f"## Checkpoints neste mapa\n\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)
        self._seen_maps.add(doom_map)
        return stem

    def _append_to_map(self, doom_map: str, ckpt_stem: str, headline: str) -> None:
        path = os.path.join(self.dir_maps, f"{self._map_stem(doom_map)}.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"- [[{ckpt_stem}]] — {headline}\n")

    # ------------------------------------------------------------------
    def _ensure_concept_note(self, name: str, description: str, created_step: int) -> str:
        """Cria a nota de conceito se for nova; retorna o stem (alvo do wikilink).

        O stem usa o nome CANÔNICO (1º visto) e a nota carrega um `id` estável,
        então variações de título do LLM não duplicam o arquivo (Passo 3).
        """
        is_new = self.registry.register(name, created_step)
        canonical = self.registry.canonical(name)
        cid = self.registry.id_for(name)
        stem = _slug_concept(canonical)
        path = os.path.join(self.dir_concept, f"{stem}.md")
        if not is_new or os.path.exists(path):
            return stem

        # Conceito genuinamente novo: pede ao LLM uma nota atemporal.
        try:
            note: ConceptNote = self.llm.generate_concept(canonical, description)
            summary, manifestation = note.summary, note.manifestation_in_doom
            related, tags = note.related, note.tags
        except Exception:
            # Fallback robusto: nunca derruba o pipeline por causa da nota.
            summary, manifestation = description, ""
            related, tags = [], ["concept"]

        related_links = "\n".join(f"- [[{_slug_concept(r)}]]" for r in related)
        fm = _yaml_frontmatter(
            {
                "type": "concept",
                "id": cid,
                "created_step": created_step,
                "tags": list(dict.fromkeys((tags or []) + ["concept"])),
            }
        )
        body = (
            f"# {canonical}\n\n"
            f"{summary}\n\n"
            f"## No treino (Doom)\n\n{manifestation}\n\n"
            f"## Relacionados\n\n{related_links}\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)
        return stem

    # ------------------------------------------------------------------
    def _fallback_checkpoint_note(
        self, snapshot: dict, previous: Optional[dict]
    ) -> CheckpointNote:
        """Nota factual quando o LLM não está disponível — o projeto funciona sem Ollama."""
        reward = snapshot.get("mean_reward", 0.0)
        acc = snapshot.get("shooting_accuracy", 0.0)
        kills = snapshot.get("kills_per_episode", 0.0)
        step = int(snapshot.get("num_timesteps", 0))
        evidence = [
            f"Recompensa média/episódio: {reward:.2f}",
            f"Precisão de tiro: {acc:.0%} "
            f"({int(snapshot.get('shots_hit', 0))}/{int(snapshot.get('shots_fired', 0))})",
            f"Kills/episódio: {kills:.2f}",
            f"Dano causado/tomado: {snapshot.get('damage_dealt', 0):.0f}/"
            f"{snapshot.get('damage_taken', 0):.0f}",
            f"Distância/episódio: {snapshot.get('distance_per_episode', 0):.0f} u",
        ]
        return CheckpointNote(
            title=f"Checkpoint @ {step:,} steps",
            headline=f"Recompensa {reward:.1f}, precisão {acc:.0%}, {kills:.1f} kills/ep.",
            behavior_change=(
                "_Resumo factual gerado SEM o LLM. Para a narrativa interpretativa, "
                "suba o `ollama serve` e rode `python -m writer.process_run`._"
            ),
            evidence=evidence,
            linked_concepts=[],
            new_concepts=[],
            tags=["sem-llm"],
        )

    def write_checkpoint(self, snapshot: dict, previous: Optional[dict]) -> str:
        """Gera e grava uma nota de checkpoint. Retorna o stem do arquivo criado."""
        try:
            note: CheckpointNote = self.llm.generate_checkpoint(
                snapshot=snapshot,
                previous=previous,
                existing_concepts=self.registry.names(),
                button_names=self.button_names,
            )
        except Exception:
            # Ollama indisponível/erro -> nota factual (não derruba o pipeline).
            note = self._fallback_checkpoint_note(snapshot, previous)

        # Blindagem: modelos pequenos às vezes despejam o relatório inteiro no
        # título/headline. Mantém só a 1ª linha, sem markdown, truncada.
        note.title = _one_line(note.title, 90) or f"Checkpoint @ {snapshot['num_timesteps']:,}"
        note.headline = _one_line(note.headline, 180) or note.title

        step = snapshot["num_timesteps"]
        self._ckpt_index += 1
        stem = f"CKPT-{self._ckpt_index:04d}-step{step}"
        path = os.path.join(self.dir_ckpt, f"{stem}.md")

        # Garante notas de conceito (existentes -> touch; novos -> cria).
        concept_links: List[str] = []
        for cname in note.linked_concepts:
            if self.registry.exists(cname):
                self.registry.touch(cname)
                concept_links.append(_slug_concept(self.registry.canonical(cname)))
            else:
                # LLM linkou algo que não existe ainda: cria com descrição mínima.
                concept_links.append(
                    self._ensure_concept_note(cname, f"Conceito referenciado em {stem}.", step)
                )
        for nc in note.new_concepts:
            concept_links.append(self._ensure_concept_note(nc.name, nc.description, step))

        # (D) Detecção de regressão: se uma métrica despencou, destaca e linka o
        # conceito "Catastrophic Forgetting" automaticamente.
        regressions = detect_regressions(snapshot, previous)
        if regressions:
            cf_stem = self._ensure_concept_note(
                FORGETTING_CONCEPT, FORGETTING_DESCRIPTION, step
            )
            concept_links.append(cf_stem)

        concept_links = list(dict.fromkeys(concept_links))  # dedup preservando ordem

        # Strip de bullet/anotações que modelos pequenos copiam do fact-sheet.
        def _clean_ev(e: str) -> str:
            e = str(e).lstrip("-•* ").strip()
            # remove a anotação "(1º checkpoint)" que o modelo copia do fact-sheet
            e = re.sub(r"\s*\((?:1º|primeiro)\s+checkpoint\)\.?", "", e, flags=re.I)
            return re.sub(r"\s{2,}", " ", e).strip()

        evidence_md = "\n".join(f"- {_clean_ev(e)}" for e in note.evidence if str(e).strip())
        concepts_md = "\n".join(f"- [[{c}]]" for c in concept_links) or "_(nenhum)_"
        prev_link = f"[[{self._last_ckpt_stem}]]" if self._last_ckpt_stem else "_(primeiro)_"
        dist = snapshot.get("action_distribution", {})
        dist_md = "\n".join(f"- `{k}`: {v:.1%}" for k, v in dist.items())

        # Modo campanha: nota do mapa + link bidirecional.
        doom_map = snapshot.get("map") or ""
        map_link = ""
        if doom_map:
            map_stem = self._ensure_map_note(doom_map)
            map_link = f"  |  **Mapa:** [[{map_stem}]]"

        fm_fields = {
            "type": "checkpoint",
            "id": stem,
            "run": self.cfg.run_name,
            "scenario": self.cfg.scenario,
            "timesteps": step,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mean_reward": round(snapshot.get("mean_reward", 0.0), 3),
            "kills_per_episode": round(snapshot.get("kills_per_episode", 0.0), 3),
            "action_entropy_norm": round(
                snapshot.get("action_entropy_normalized", 0.0), 3
            ),
            "shooting_accuracy": round(snapshot.get("shooting_accuracy", 0.0), 3),
            "distance_per_episode": round(snapshot.get("distance_per_episode", 0.0), 1),
            "tags": list(dict.fromkeys((note.tags or []) + ["checkpoint", "doom-rl"])),
        }
        if doom_map:
            fm_fields["map"] = doom_map
            fm_fields["success_rate"] = round(snapshot.get("success_rate", 0.0), 3)
        if regressions:
            fm_fields["regression"] = True
            fm_fields["tags"] = list(dict.fromkeys(fm_fields["tags"] + ["regression"]))
        fm = _yaml_frontmatter(fm_fields)

        # Callout de regressão (Obsidian renderiza `> [!warning]`).
        regression_md = ""
        if regressions:
            bullets = "\n".join(f"> - {r}" for r in regressions)
            regression_md = (
                "> [!warning] Regressão detectada\n"
                f"> Possível esquecimento — ver [[{_slug_concept(FORGETTING_CONCEPT)}]].\n"
                f"{bullets}\n\n"
            )

        success_md = ""
        if doom_map:
            success_md = (
                f"## Progresso no mapa {doom_map}\n\n"
                f"- Taxa de conclusão (sucesso): "
                f"{snapshot.get('success_rate', 0.0):.1%}\n\n"
            )

        # Pontaria (acertos x erros) e exploração do mapa (caminho percorrido).
        cov = snapshot.get("map_coverage", {}) or {}
        weapons = snapshot.get("weapons_used", {}) or {}
        weapons_md = (
            ", ".join(f"`{k}` {v:.0%}" for k, v in weapons.items()) or "_(n/d)_"
        )
        aim_md = (
            "## Pontaria e exploração\n\n"
            f"- Tiros: {snapshot.get('shots_fired', 0):.0f} disparados, "
            f"{snapshot.get('shots_hit', 0):.0f} acertos, "
            f"{snapshot.get('shots_missed', 0):.0f} erros "
            f"(**precisão {snapshot.get('shooting_accuracy', 0.0):.0%}**)\n"
            f"- Caminho: {snapshot.get('distance_traveled', 0.0):,.0f} u no total, "
            f"{snapshot.get('distance_per_episode', 0.0):,.0f} u/episódio\n"
            f"- Exploração: {int(snapshot.get('cells_visited', 0))} células visitadas "
            f"(~{cov.get('explored_fraction', 0.0):.0%} da área percorrida)\n"
            f"- Armas usadas: {weapons_md}\n\n"
        )

        # Minimapa do caminho percorrido (heatmap de visitas) -> attachments/.
        minimap_md = ""
        path_cells = snapshot.get("path_cells") or []
        if path_cells:
            img_name = f"{stem}.png"
            try:
                if render_minimap(
                    path_cells,
                    os.path.join(self.dir_attach, img_name),
                    walls=snapshot.get("map_walls"),
                ):
                    minimap_md = (
                        "## Minimapa do nível (caminho percorrido)\n\n"
                        f"![[{img_name}]]\n\n"
                        "_Paredes reais do mapa; mais quente = onde o agente passou "
                        "mais tempo._\n\n"
                    )
            except Exception:
                minimap_md = ""  # imagem é opcional; nunca derruba a nota

        body = (
            f"# {note.title}\n\n"
            f"> {note.headline}\n\n"
            f"**Run:** [[{self.cfg.run_name}]]  |  **Anterior:** {prev_link}"
            f"{map_link}  |  **Timesteps:** {step:,}\n\n"
            f"{regression_md}"
            f"{success_md}"
            f"## O que mudou no comportamento\n\n{note.behavior_change}\n\n"
            f"## Evidências\n\n{evidence_md}\n\n"
            f"{aim_md}"
            f"{minimap_md}"
            f"## Distribuição de ações (entropia norm. "
            f"{snapshot.get('action_entropy_normalized', 0.0):.2f})\n\n{dist_md}\n\n"
            f"## Conceitos de RL\n\n{concepts_md}\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)

        self._append_to_run(stem, note.headline)
        if doom_map:
            self._append_to_map(doom_map, stem, note.headline)
        self._last_ckpt_stem = stem
        return stem
