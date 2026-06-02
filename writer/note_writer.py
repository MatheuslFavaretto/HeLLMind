"""Writes the .md notes into the vault (checkpoints, concepts, runs, maps).

Obsidian reads .md files directly — no plugin or API needed. Wikilinks [[Name]]
reference the file STEM (without .md).
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

# Below this many episodes in a window, the signal is too noisy for strong claims.
LOW_SIGNAL_EPISODES = 3


def _one_line(text: str, max_len: int) -> str:
    """First line, no header/quote markdown, truncated — for title/headline."""
    s = str(text or "").strip()
    s = s.split("\n", 1)[0].lstrip("#> ").strip()
    return s[:max_len].strip()


def _strip_citations(text: str) -> str:
    """Remove fake citations small models invent: '(Report, p. ...)', '(2)'."""
    s = str(text or "")
    s = re.sub(r"\s*\((?:Report|Relat[óo]rio)[^)]*\)", "", s, flags=re.I)
    s = re.sub(r"\s*\(\d{1,2}\)", "", s)  # standalone numeric ref markers (not "(25%)")
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def _slug_concept(name: str) -> str:
    """File name of a concept note (= wikilink target, without .md)."""
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
        self.llm = LLMWriter(
            model=cfg.llm_model,
            host=cfg.ollama_host,
            num_ctx=cfg.llm_num_ctx,
            num_predict=cfg.llm_num_predict,
            keep_alive=cfg.llm_keep_alive,
        )

        self.dir_index = os.path.join(cfg.vault_path, cfg.dir_index)
        self.dir_ckpt = os.path.join(cfg.vault_path, cfg.dir_checkpoints)
        self.dir_concept = os.path.join(cfg.vault_path, cfg.dir_concepts)
        self.dir_runs = os.path.join(cfg.vault_path, cfg.dir_runs)
        self.dir_maps = os.path.join(cfg.vault_path, cfg.dir_maps)
        self.dir_lessons = os.path.join(cfg.vault_path, cfg.dir_lessons)
        self.dir_attach = os.path.join(cfg.vault_path, cfg.dir_attachments)
        for d in (self.dir_index, self.dir_ckpt, self.dir_concept, self.dir_runs,
                  self.dir_maps, self.dir_attach):
            os.makedirs(d, exist_ok=True)

        self.registry = ConceptRegistry(
            os.path.join(cfg.vault_path, ".concept_registry.json")
        )
        self._ckpt_index = 0
        self._last_ckpt_stem: Optional[str] = None
        self._seen_maps: set = set()
        self._last_walls: list = []  # geometry is logged once per map; remember it
        self._ensure_run_note()

    # ------------------------------------------------------------------
    def _task_label(self) -> str:
        """Correct task description (campaign vs scenario) for the notes."""
        if self.cfg.campaign:
            wad = os.path.basename(self.cfg.wad_path) or "WAD"
            return f"campaign · {wad} · maps {', '.join(self.cfg.maps)}"
        return f"scenario {self.cfg.scenario}"

    def _ensure_run_note(self) -> None:
        path = os.path.join(self.dir_runs, f"{self.cfg.run_name}.md")
        if os.path.exists(path):
            return
        fm = _yaml_frontmatter(
            {
                "type": "run",
                "mode": "campaign" if self.cfg.campaign else "scenario",
                "task": self._task_label(),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tags": ["run", "doom-rl"],
            }
        )
        body = (
            f"# Run: {self.cfg.run_name}\n\n"
            f"PPO training — **{self._task_label()}**.\n\n"
            f"## Checkpoints\n\n"
            f"_Checkpoint notes for this run appear here as training progresses._\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)

    def _append_to_run(self, ckpt_stem: str, headline: str) -> None:
        path = os.path.join(self.dir_runs, f"{self.cfg.run_name}.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"- [[{ckpt_stem}]] — {headline}\n")

    def write_run_chart(self, snapshots: List[dict]) -> Optional[str]:
        """Render the run's learning curve and embed it in the run note."""
        from writer.charts import render_learning_curve

        img_name = f"{self.cfg.run_name}-curve.png"
        out = os.path.join(self.dir_attach, img_name)
        if not render_learning_curve(snapshots, out):
            return None
        path = os.path.join(self.dir_runs, f"{self.cfg.run_name}.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## Learning curve\n\n![[{img_name}]]\n")
        return img_name

    def write_run_minimap(self, snapshots: List[dict]) -> Optional[str]:
        """Embed the level map (walls + path) right in the run note, so the run's main
        page shows WHERE it played — not only the per-checkpoint notes. Uses the last
        snapshot that has a path, plus the cross-run exploration memory for the map."""
        snap = next((s for s in reversed(snapshots) if s.get("path_cells")), None)
        if snap is None:
            return None
        walls = snap.get("map_walls") or self._last_walls
        memory_cells = None
        if self.cfg.campaign and snap.get("map"):
            try:
                from writer.coverage_store import CoverageStore
                memory_cells = CoverageStore(self.cfg.memory_dir).load_cells(snap["map"])
            except Exception:
                memory_cells = None
        img_name = f"{self.cfg.run_name}-map.png"
        try:
            ok = render_minimap(snap["path_cells"], os.path.join(self.dir_attach, img_name),
                                walls=walls, polyline=snap.get("path_polyline"),
                                memory_cells=memory_cells)
        except Exception:
            ok = False
        if not ok:
            return None
        path = os.path.join(self.dir_runs, f"{self.cfg.run_name}.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## Level map (where it played)\n\n![[{img_name}]]\n"
                    "\n_Real walls; hotter = more time; blue line = one episode "
                    "(🟢→🔴); dim teal = explored across runs._\n")
        return img_name

    def write_run_story(self, snapshots: List[dict]) -> Optional[str]:
        """(A) Narrative synthesis of the whole run; linked from the run note."""
        if not snapshots:
            return None

        # The LLM tells the arc; if it fails, fall back to a minimal factual summary.
        try:
            story = self.llm.generate_run_story(
                self.cfg.run_name, snapshots, self.registry.names()
            )
            title, narrative = story.title, story.narrative
            milestones, key_concepts = story.milestones, story.key_concepts
        except Exception:
            first, last = snapshots[0], snapshots[-1]
            title = f"Synthesis: {self.cfg.run_name}"
            narrative = (
                f"Run with {len(snapshots)} checkpoints, from "
                f"{int(first.get('num_timesteps', 0)):,} to "
                f"{int(last.get('num_timesteps', 0)):,} steps. "
                f"Shooting accuracy went from {first.get('shooting_accuracy', 0):.0%} "
                f"to {last.get('shooting_accuracy', 0):.0%}; mean reward from "
                f"{first.get('mean_reward', 0):.1f} to {last.get('mean_reward', 0):.1f}."
            )
            milestones, key_concepts = [], []

        # Link existing concepts by canonical name (matched by stable slug id).
        concept_links = [
            _slug_concept(self.registry.canonical(c))
            for c in key_concepts
            if self.registry.exists(c)
        ]
        # Fallback: if nothing matched, link the most-mentioned concepts of the vault
        # so the synthesis is never an orphan node.
        if not concept_links:
            concept_links = [_slug_concept(n) for n in self.registry.top(6)]
        concepts_md = "\n".join(f"- [[{c}]]" for c in dict.fromkeys(concept_links)) or "_(none)_"
        milestones_md = "\n".join(f"- {m}" for m in milestones) or "_(n/a)_"

        stem = f"{self.cfg.run_name} - Synthesis"
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
            f"## Milestones\n\n{milestones_md}\n\n"
            f"## Core concepts\n\n{concepts_md}\n"
        )
        with open(os.path.join(self.dir_runs, f"{stem}.md"), "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)

        run_path = os.path.join(self.dir_runs, f"{self.cfg.run_name}.md")
        with open(run_path, "a", encoding="utf-8") as f:
            f.write(f"\n## Synthesis\n\n- [[{stem}]]\n")
        return stem

    # ------------------------------------------------------------------
    def _stems_in(self, directory: str, exclude: str = "") -> List[str]:
        """Wikilink stems of the .md notes in a vault folder (for the MOC hub)."""
        if not os.path.isdir(directory):
            return []
        out = []
        for fname in sorted(os.listdir(directory)):
            if fname.endswith(".md"):
                stem = fname[:-3]
                if not exclude or exclude not in stem:
                    out.append(stem)
        return out

    def write_knowledge_hub(self) -> str:
        """(Phase 2) A Map-of-Content note that connects runs, maps, concepts and
        lessons — so the Graph View becomes one connected network, not islands."""
        runs = [s for s in self._stems_in(self.dir_runs) if not s.endswith("- Synthesis")]
        syntheses = [s for s in self._stems_in(self.dir_runs) if s.endswith("- Synthesis")]
        maps = self._stems_in(self.dir_maps)
        concepts = [_slug_concept(n) for n in self.registry.top(12)]
        lessons_exists = os.path.exists(os.path.join(self.dir_lessons, "Lessons.md"))

        def section(title, stems):
            if not stems:
                return ""
            links = "\n".join(f"- [[{s}]]" for s in stems)
            return f"## {title}\n\n{links}\n\n"

        fm = _yaml_frontmatter({
            "type": "moc",
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tags": ["moc", "index", "doom-rl"],
        })
        body = (
            "# 🔥 HeLLMind — Knowledge Graph\n\n"
            "Hub connecting everything the agent has learned and documented.\n\n"
            + section("Runs", runs)
            + section("Run syntheses", syntheses)
            + section("Maps", maps)
            + section("Core concepts", concepts)
            + ("## Lessons\n\n- [[Lessons]]\n\n" if lessons_exists else "")
            + ("## Control\n\n- [[control]] — edit to steer training live\n" )
        )
        path = os.path.join(self.dir_index, "Knowledge Graph.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)
        return path

    # ------------------------------------------------------------------
    def _map_stem(self, doom_map: str) -> str:
        """Stem (wikilink target) of a map note, e.g. 'Map - MAP01'."""
        return f"Map - {doom_map}"

    def _ensure_map_note(self, doom_map: str) -> str:
        """Create the map note (one per map) if missing; return the stem."""
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
            f"# Map: {doom_map}\n\n"
            f"Agent progress on map **{doom_map}** (run [[{self.cfg.run_name}]]).\n\n"
            f"## Checkpoints on this map\n\n"
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
    def _ensure_concept_note(
        self, name: str, description: str, created_step: int, use_llm: bool = True
    ) -> str:
        """Create the concept note if new; return the stem (wikilink target).

        The stem uses the CANONICAL name (first seen) and the note carries a stable
        `id`, so small title variations from the LLM don't duplicate the file.

        `use_llm=False` writes a cheap stub (no Ollama call) — used to cap LLM cost
        for referenced/overflow concepts; the stub can be enriched on a later run.
        """
        is_new = self.registry.register(name, created_step)
        canonical = self.registry.canonical(name)
        cid = self.registry.id_for(name)
        stem = _slug_concept(canonical)
        path = os.path.join(self.dir_concept, f"{stem}.md")
        if not is_new or os.path.exists(path):
            return stem

        # Genuinely new concept: ask the LLM for a timeless note (unless capped).
        if use_llm:
            try:
                note: ConceptNote = self.llm.generate_concept(canonical, description)
                summary, manifestation = note.summary, note.manifestation_in_doom
                related, tags = note.related, note.tags
            except Exception:
                # Robust fallback: a note never crashes the pipeline.
                summary, manifestation = description, ""
                related, tags = [], ["concept"]
        else:
            summary, manifestation = description, ""
            related, tags = [], ["concept", "stub"]

        # Only link related concepts that already exist (avoid orphan/grey links).
        related_links = "\n".join(
            f"- [[{_slug_concept(self.registry.canonical(r))}]]"
            for r in related
            if self.registry.exists(r)
        ) or "_(none yet)_"
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
            f"## In training (Doom)\n\n{manifestation}\n\n"
            f"## Related\n\n{related_links}\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)
        return stem

    # ------------------------------------------------------------------
    def _fallback_checkpoint_note(
        self, snapshot: dict, previous: Optional[dict]
    ) -> CheckpointNote:
        """Factual note when the LLM is unavailable — the project works without Ollama."""
        reward = snapshot.get("mean_reward", 0.0)
        acc = snapshot.get("shooting_accuracy", 0.0)
        kills = snapshot.get("kills_per_episode", 0.0)
        step = int(snapshot.get("num_timesteps", 0))
        evidence = [
            f"Mean reward/episode: {reward:.2f}",
            f"Shooting accuracy: {acc:.0%} "
            f"({int(snapshot.get('shots_hit', 0))}/{int(snapshot.get('shots_fired', 0))})",
            f"Kills/episode: {kills:.2f}",
            f"Damage dealt/taken: {snapshot.get('damage_dealt', 0):.0f}/"
            f"{snapshot.get('damage_taken', 0):.0f}",
            f"Distance/episode: {snapshot.get('distance_per_episode', 0):.0f} u",
        ]
        return CheckpointNote(
            title=f"Checkpoint @ {step:,} steps",
            headline=f"Reward {reward:.1f}, accuracy {acc:.0%}, {kills:.1f} kills/ep.",
            behavior_change=(
                "_Factual summary generated WITHOUT the LLM. For the interpretive "
                "narrative, start `ollama serve` and run `python -m writer.process_run`._"
            ),
            evidence=evidence,
            linked_concepts=[],
            new_concepts=[],
            tags=["no-llm"],
        )

    def write_checkpoint(self, snapshot: dict, previous: Optional[dict]) -> str:
        """Generate and write a checkpoint note. Returns the created file stem."""
        try:
            note: CheckpointNote = self.llm.generate_checkpoint(
                snapshot=snapshot,
                previous=previous,
                existing_concepts=self.registry.names(),
                button_names=self.button_names,
            )
        except Exception:
            # Ollama unavailable/error -> factual note (never crash the pipeline).
            note = self._fallback_checkpoint_note(snapshot, previous)

        # Hardening: small models sometimes dump the whole report into the
        # title/headline. Keep only the first line, no markdown, truncated.
        note.title = _one_line(note.title, 90) or f"Checkpoint @ {snapshot['num_timesteps']:,}"
        note.headline = _one_line(note.headline, 180) or note.title
        note.behavior_change = _strip_citations(note.behavior_change)

        step = snapshot["num_timesteps"]
        self._ckpt_index += 1
        stem = f"CKPT-{self._ckpt_index:04d}-step{step}"
        path = os.path.join(self.dir_ckpt, f"{stem}.md")

        # Ensure concept notes (existing -> touch; new -> create). To cap Ollama
        # cost, only the first few NEW concepts get an LLM-written body; the rest
        # (and linked-but-missing ones) are cheap stubs, enriched on a later run.
        concept_links: List[str] = []
        llm_budget = self.cfg.max_new_concepts_per_ckpt
        for cname in note.linked_concepts:
            if self.registry.exists(cname):
                self.registry.touch(cname)
                concept_links.append(_slug_concept(self.registry.canonical(cname)))
            else:
                # Linked something that doesn't exist yet: cheap stub.
                concept_links.append(
                    self._ensure_concept_note(
                        cname, f"Concept referenced in {stem}.", step, use_llm=False
                    )
                )
        for nc in note.new_concepts:
            use_llm = llm_budget > 0
            if use_llm:
                llm_budget -= 1
            concept_links.append(
                self._ensure_concept_note(nc.name, nc.description, step, use_llm=use_llm)
            )

        # (D) Regression detection: if a metric dropped sharply, highlight it and link
        # the "Catastrophic Forgetting" concept automatically.
        regressions = detect_regressions(snapshot, previous)
        if regressions:
            cf_stem = self._ensure_concept_note(
                FORGETTING_CONCEPT, FORGETTING_DESCRIPTION, step
            )
            concept_links.append(cf_stem)

        concept_links = list(dict.fromkeys(concept_links))  # dedup, keep order

        # Strip bullets/annotations that small models copy from the fact sheet.
        def _clean_ev(e: str) -> str:
            e = str(e).lstrip("-•* ").strip()
            e = re.sub(r"\s*\((?:1st|first|1º|primeiro)\s+checkpoint\)\.?", "", e, flags=re.I)
            e = _strip_citations(e)
            return re.sub(r"\s{2,}", " ", e).strip()

        evidence_md = "\n".join(f"- {_clean_ev(e)}" for e in note.evidence if str(e).strip())
        concepts_md = "\n".join(f"- [[{c}]]" for c in concept_links) or "_(none)_"
        prev_link = f"[[{self._last_ckpt_stem}]]" if self._last_ckpt_stem else "_(first)_"
        dist = snapshot.get("action_distribution", {})
        dist_md = "\n".join(f"- `{k}`: {v:.1%}" for k, v in dist.items())

        # Campaign mode: map note + bidirectional link.
        doom_map = snapshot.get("map") or ""
        map_link = ""
        if doom_map:
            map_stem = self._ensure_map_note(doom_map)
            map_link = f"  |  **Map:** [[{map_stem}]]"

        low_signal = int(snapshot.get("episodes", 0)) < LOW_SIGNAL_EPISODES

        fm_fields = {
            "type": "checkpoint",
            "id": stem,
            "run": self.cfg.run_name,
            "mode": "campaign" if self.cfg.campaign else "scenario",
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
        if self.cfg.campaign:
            if doom_map:
                fm_fields["map"] = doom_map
                fm_fields["success_rate"] = round(snapshot.get("success_rate", 0.0), 3)
        else:
            fm_fields["scenario"] = self.cfg.scenario
        if low_signal:
            fm_fields["low_signal"] = True
        if regressions:
            fm_fields["regression"] = True
            fm_fields["tags"] = list(dict.fromkeys(fm_fields["tags"] + ["regression"]))
        fm = _yaml_frontmatter(fm_fields)

        # Low-signal caveat (few episodes -> noisy; don't over-read the narrative).
        low_signal_md = ""
        if low_signal:
            low_signal_md = (
                f"> [!note] Low signal — only {int(snapshot.get('episodes', 0))} "
                "episode(s) in this window; treat the narrative as tentative.\n\n"
            )

        # Regression callout (Obsidian renders `> [!warning]`).
        regression_md = ""
        if regressions:
            bullets = "\n".join(f"> - {r}" for r in regressions)
            regression_md = (
                "> [!warning] Regression detected\n"
                f"> Possible forgetting — see [[{_slug_concept(FORGETTING_CONCEPT)}]].\n"
                f"{bullets}\n\n"
            )

        success_md = ""
        if doom_map:
            success_md = (
                f"## Map progress ({doom_map})\n\n"
                f"- Completion (success) rate: "
                f"{snapshot.get('success_rate', 0.0):.1%}\n\n"
            )

        # Aim (hits vs misses) and map exploration (path taken).
        cov = snapshot.get("map_coverage", {}) or {}
        weapons = snapshot.get("weapons_used", {}) or {}
        weapons_md = (
            ", ".join(f"`{k}` {v:.0%}" for k, v in weapons.items()) or "_(n/a)_"
        )
        aim_md = (
            "## Aim & exploration\n\n"
            f"- Shots: {snapshot.get('shots_fired', 0):.0f} fired, "
            f"{snapshot.get('shots_hit', 0):.0f} hits, "
            f"{snapshot.get('shots_missed', 0):.0f} misses "
            f"(**accuracy {snapshot.get('shooting_accuracy', 0.0):.0%}**)\n"
            f"- Path: {snapshot.get('distance_traveled', 0.0):,.0f} u total, "
            f"{snapshot.get('distance_per_episode', 0.0):,.0f} u/episode\n"
            f"- Exploration: {int(snapshot.get('cells_visited', 0))} cells visited "
            f"(~{cov.get('explored_fraction', 0.0):.0%} of the traversed area)\n"
            f"- Weapons used: {weapons_md}\n\n"
        )

        # Level minimap with the path taken (heatmap) -> attachments/.
        minimap_md = ""
        path_cells = snapshot.get("path_cells") or []
        # Walls are logged once per map; reuse the last seen ones when deduped.
        if snapshot.get("map_walls"):
            self._last_walls = snapshot["map_walls"]
        if path_cells:
            img_name = f"{stem}.png"
            # Cross-run exploration memory for THIS map (faint background layer): what the
            # agent has explored over every past run, so the minimap shows accumulated
            # knowledge, not just this window. Campaign only (maps have stable geometry).
            memory_cells = None
            if self.cfg.campaign and snapshot.get("map"):
                try:
                    from writer.coverage_store import CoverageStore
                    memory_cells = CoverageStore(self.cfg.memory_dir).load_cells(
                        snapshot["map"])
                except Exception:
                    memory_cells = None
            try:
                if render_minimap(
                    path_cells,
                    os.path.join(self.dir_attach, img_name),
                    walls=self._last_walls,
                    polyline=snapshot.get("path_polyline"),
                    memory_cells=memory_cells,
                ):
                    mem_note = (
                        " Dim teal = explored in past runs (persistent memory)."
                        if memory_cells else ""
                    )
                    minimap_md = (
                        "## Level minimap (path taken)\n\n"
                        f"![[{img_name}]]\n\n"
                        "_Real level walls; hotter = more time spent. The blue line is one "
                        f"episode's route (🟢 start → 🔴 end).{mem_note}_\n\n"
                    )
            except Exception:
                minimap_md = ""  # image is optional; never crash the note

        body = (
            f"# {note.title}\n\n"
            f"> {note.headline}\n\n"
            f"**Run:** [[{self.cfg.run_name}]]  |  **Previous:** {prev_link}"
            f"{map_link}  |  **Timesteps:** {step:,}\n\n"
            f"{low_signal_md}"
            f"{regression_md}"
            f"{success_md}"
            f"## What changed in behavior\n\n{note.behavior_change}\n\n"
            f"## Evidence\n\n{evidence_md}\n\n"
            f"{aim_md}"
            f"{minimap_md}"
            f"## Action distribution (norm. entropy "
            f"{snapshot.get('action_entropy_normalized', 0.0):.2f})\n\n{dist_md}\n\n"
            f"## RL concepts\n\n{concepts_md}\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + body)

        self._append_to_run(stem, note.headline)
        if doom_map:
            self._append_to_map(doom_map, stem, note.headline)
        self._last_ckpt_stem = stem
        return stem
