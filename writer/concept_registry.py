"""Registro de conceitos de RL já criados no vault, com IDs determinísticos.

Por que IDs (Passo 3): confiar que o LLM repita o nome EXATO de um conceito gera
duplicatas e links quebrados ("Agressivo" vs "agressivo " vs "Reward  Shaping").
Aqui cada conceito tem um `id` estável derivado de um slug normalizado (sem acento,
minúsculo, só [a-z0-9_]). O Python casa os conceitos por ESSE id, não pelo título
que o LLM digitou — então variações de caixa/acentuação/espaço colapsam no mesmo
arquivo, e o nome canônico (o primeiro visto) é preservado.

Estrutura interna: { id -> {"name": <canônico>, "created_step": int, "mentions": int} }
"""
import json
import os
import re
import unicodedata
from typing import Dict, List


def clean_concept_name(name: str) -> str:
    """Limpa o nome que o LLM sugeriu (modelos pequenos alucinam URLs/markdown).

    Ex.: 'Exploration vs Exploitation https://obsidian.md/notes/...' -> 'Exploration
    vs Exploitation'. Corta na 1ª URL/linha/markdown e trunca, para não poluir o
    nome do arquivo nem o wikilink.
    """
    s = (name or "").strip()
    s = s.split("\n", 1)[0]  # só a primeira linha
    s = s.replace("[[", "").replace("]]", "")  # mantém o texto do wikilink
    # corta no começo de uma URL (o LLM cola links no nome)
    s = re.split(r"https?://|www\.|\]\(|\(http", s, maxsplit=1, flags=re.I)[0]
    s = re.sub(r"[#*`_>~|/\\\[\]()<>{}]+", " ", s)  # remove ruído de markdown/barras
    s = re.sub(r"\s+", " ", s).strip(" -:;,.\"'")
    return s[:48].strip() or "Conceito"


def concept_id(name: str) -> str:
    """Slug estável -> id do conceito. 'Reward Shaping' == 'reward  shaping'."""
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
        # Migra formatos antigos (chaveados pelo nome) para o formato por id.
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
        """Nomes canônicos (o que mostramos ao LLM como 'conceitos existentes')."""
        return [v["name"] for v in sorted(self._data.values(), key=lambda d: d["name"])]

    def exists(self, name: str) -> bool:
        return concept_id(name) in self._data

    def id_for(self, name: str) -> str:
        return concept_id(name)

    def canonical(self, name: str) -> str:
        """Nome canônico (1º visto) para este conceito; o título do LLM se varia."""
        cid = concept_id(name)
        return self._data[cid]["name"] if cid in self._data else name

    def register(self, name: str, created_step: int) -> bool:
        """Registra um conceito. Retorna True se foi criado AGORA (id inédito)."""
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
