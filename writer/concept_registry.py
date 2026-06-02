"""Registry of RL concepts already created in the vault, with deterministic IDs.

Why IDs: trusting the LLM to repeat the EXACT concept name produces duplicates and
broken links ("Aggressive" vs "aggressive " vs "Reward  Shaping"). Here each concept
has a stable `id` derived from a normalized slug (no accents, lowercase, [a-z0-9_]).
Python matches concepts by THAT id, not by the title the LLM typed — so case/accent/
whitespace variations collapse into the same file, and the canonical name (first seen)
is preserved.

Internal structure: { id -> {"name": <canonical>, "created_step": int, "mentions": int} }
"""
import json
import os
import re
import unicodedata
from typing import Dict, List


def clean_concept_name(name: str) -> str:
    """Clean the name the LLM suggested (small models hallucinate URLs/markdown).

    e.g. 'Exploration vs Exploitation https://obsidian.md/notes/...' -> 'Exploration
    vs Exploitation'. Cuts at the first URL/line/markdown and truncates, so it doesn't
    pollute the file name or the wikilink.
    """
    s = (name or "").strip()
    s = s.split("\n", 1)[0]  # first line only
    s = s.replace("[[", "").replace("]]", "")  # keep the wikilink text
    # cut at the start of a URL (the LLM glues links onto the name)
    s = re.split(r"https?://|www\.|\]\(|\(http", s, maxsplit=1, flags=re.I)[0]
    s = re.sub(r"[#*`_>~|/\\\[\]()<>{}]+", " ", s)  # strip markdown/slash noise
    s = re.sub(r"\s+", " ", s).strip()
    # Small models append trend/value tails to the name ("Action Entropy down from
    # 09 to 07"). Keep only the canonical term so duplicates collapse to one note.
    s = re.split(
        r"\s+(?:up|down|increased|decreased|rose|fell|dropped|rising|falling|"
        r"higher|lower)\b",
        s, maxsplit=1, flags=re.I,
    )[0]
    s = re.split(r"\s+\d", s, maxsplit=1)[0]  # cut before a number ("Accuracy 25")
    s = re.sub(r"\s+", " ", s).strip(" -:;,.\"'")
    return s[:48].strip() or "Concept"


def concept_id(name: str) -> str:
    """Stable slug -> concept id. 'Reward Shaping' == 'reward  shaping'."""
    s = clean_concept_name(name)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return f"concept_{s}" if s else "concept_unknown"


class ConceptRegistry:
    def __init__(self, registry_path: str) -> None:
        self.path = registry_path
        self._data: Dict[str, dict] = {}  # id -> {name, created_step, mentions}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Migrate old (name-keyed) formats to the id-keyed format.
        migrated: Dict[str, dict] = {}
        for key, val in raw.items():
            name = clean_concept_name(val.get("name", key))
            cid = concept_id(name)
            migrated[cid] = {
                "name": name,
                "created_step": int(val.get("created_step", 0)),
                "mentions": int(val.get("mentions", 1)),
            }
        self._data = migrated

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    def names(self) -> List[str]:
        """Canonical names (what we show the LLM as 'existing concepts')."""
        return [v["name"] for v in sorted(self._data.values(), key=lambda d: d["name"])]

    def top(self, n: int = 6) -> List[str]:
        """Most-mentioned canonical names (used as a synthesis link fallback)."""
        items = sorted(self._data.values(), key=lambda d: (-d["mentions"], d["name"]))
        return [v["name"] for v in items[:n]]

    def exists(self, name: str) -> bool:
        return concept_id(name) in self._data

    def id_for(self, name: str) -> str:
        return concept_id(name)

    def canonical(self, name: str) -> str:
        """Canonical name (first seen) for this concept; the LLM's title if unseen."""
        cid = concept_id(name)
        return self._data[cid]["name"] if cid in self._data else name

    def register(self, name: str, created_step: int) -> bool:
        """Register a concept. Returns True if it was created NOW (new id)."""
        cid = concept_id(name)
        if cid in self._data:
            self._data[cid]["mentions"] += 1
            self._save()
            return False
        self._data[cid] = {
            "name": clean_concept_name(name),
            "created_step": int(created_step),
            "mentions": 1,
        }
        self._save()
        return True

    def touch(self, name: str) -> None:
        cid = concept_id(name)
        if cid in self._data:
            self._data[cid]["mentions"] += 1
            self._save()
