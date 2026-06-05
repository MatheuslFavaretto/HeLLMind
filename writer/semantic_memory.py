"""Semantic memory — vector DB for episodic recall (V2 Phase 3).

Stores episode events as dense embeddings so recall("died to revenant near door") can
surface similar past situations by MEANING, not just keyword match. Complements the
structured SQLite store (writer.db) which handles exact queries.

Storage: SQLite + embeddings as BLOB (float32 arrays). No new service needed.
Embeddings: nomic-embed-text via the local Ollama model (already in requirements).
Fallback: TF-IDF cosine similarity (numpy only, no Ollama needed) — slower but works
          offline.

Usage:
    from writer.semantic_memory import SemanticMemory
    mem = SemanticMemory(memory_dir)
    mem.add("agent died at low HP in corridor near DoomImp", meta={"map": "MAP01"})
    results = mem.search("low health death near enemy", top_k=5)
    # returns [(text, meta, score), ...]

    doom-cli semantic recall "deaths near corridor"
    doom-cli semantic stats
"""
from __future__ import annotations

import json
import os
import sqlite3
import struct
from typing import Any, Optional

import numpy as np


_EMBED_DIM_DEFAULT = 768   # nomic-embed-text output dim
_DB_FILENAME = "semantic_memory.db"


# ── Embedding backends ────────────────────────────────────────────────────────

def _embed_ollama(texts: list[str], model: str = "nomic-embed-text",
                  host: str = "http://localhost:11434") -> Optional[np.ndarray]:
    """Embed a batch of texts with Ollama. Returns (N, dim) float32 or None on error."""
    try:
        from ollama import Client
        client = Client(host=host)
        vecs = [client.embeddings(model=model, prompt=t).embedding for t in texts]
        return np.array(vecs, dtype=np.float32)
    except Exception:
        return None


def _embed_tfidf(texts: list[str],
                 vocab: Optional[dict[str, int]] = None) -> tuple[np.ndarray, dict[str, int]]:
    """Minimal TF-IDF embedding (numpy only). Returns (vecs, vocab)."""
    import re
    tokenise = lambda s: re.findall(r"\b[a-z]+\b", s.lower())
    if vocab is None:
        terms: set[str] = set()
        for t in texts:
            terms.update(tokenise(t))
        vocab = {w: i for i, w in enumerate(sorted(terms))}

    n, d = len(texts), len(vocab)
    tf = np.zeros((n, d), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = tokenise(t)
        for tok in toks:
            if tok in vocab:
                tf[i, vocab[tok]] += 1.0
        if toks:
            tf[i] /= len(toks)
    # IDF over this batch
    df = (tf > 0).sum(axis=0).astype(np.float32) + 1.0
    idf = np.log(n / df + 1.0)
    vecs = tf * idf
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return (vecs / norms), vocab


# ── Semantic Memory ───────────────────────────────────────────────────────────

class SemanticMemory:
    """Persistent vector store backed by SQLite (no external service required)."""

    def __init__(self, memory_dir: str,
                 embed_model: str = "nomic-embed-text",
                 ollama_host: str = "http://localhost:11434") -> None:
        os.makedirs(memory_dir, exist_ok=True)
        self._db_path = os.path.join(memory_dir, _DB_FILENAME)
        self._model = embed_model
        self._host  = ollama_host
        self._use_ollama: Optional[bool] = None  # lazily probe
        self._tfidf_vocab: dict[str, int] = {}
        self._conn = sqlite3.connect(self._db_path)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                text     TEXT NOT NULL,
                meta     TEXT,          -- JSON
                embedding BLOB          -- float32 little-endian
            )
        """)
        self._conn.commit()

    def _can_ollama(self) -> bool:
        if self._use_ollama is not None:
            return self._use_ollama
        try:
            from ollama import Client
            client = Client(host=self._host)
            models = [m.model for m in client.list().models]
            wanted = self._model if ":" in self._model else self._model + ":latest"
            self._use_ollama = any((m or "") == wanted for m in models)
        except Exception:
            self._use_ollama = False
        return self._use_ollama

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts → normalised (N, dim) float32."""
        if self._can_ollama():
            vecs = _embed_ollama(texts, self._model, self._host)
            if vecs is not None:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
                return vecs / norms
        # Fallback: TF-IDF over the entire corpus + new texts for consistent vocab
        all_rows = self._conn.execute("SELECT text FROM entries").fetchall()
        corpus = [r[0] for r in all_rows] + texts
        vecs, vocab = _embed_tfidf(corpus)
        self._tfidf_vocab = vocab
        return vecs[len(all_rows):]  # only the new texts

    def add(self, text: str, meta: Optional[dict[str, Any]] = None) -> int:
        """Embed and store one entry. Returns the new row id."""
        vec = self._embed([text])[0]
        blob = struct.pack(f"{len(vec)}f", *vec.tolist())
        cur = self._conn.execute(
            "INSERT INTO entries (text, meta, embedding) VALUES (?,?,?)",
            (text, json.dumps(meta or {}), blob))
        self._conn.commit()
        return cur.lastrowid

    def add_batch(self, texts: list[str],
                  metas: Optional[list[dict[str, Any]]] = None) -> list[int]:
        """Embed and store a batch (one Ollama round-trip)."""
        metas = metas or [{} for _ in texts]
        vecs  = self._embed(texts)
        ids   = []
        for text, meta, vec in zip(texts, metas, vecs):
            blob = struct.pack(f"{len(vec)}f", *vec.tolist())
            cur  = self._conn.execute(
                "INSERT INTO entries (text, meta, embedding) VALUES (?,?,?)",
                (text, json.dumps(meta), blob))
            ids.append(cur.lastrowid)
        self._conn.commit()
        return ids

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, dict, float]]:
        """Return top_k (text, meta, cosine_score) most similar to query."""
        rows = self._conn.execute(
            "SELECT id, text, meta, embedding FROM entries").fetchall()
        if not rows:
            return []

        if self._can_ollama():
            # Stored embeddings are already normalised Ollama vecs — just decode them.
            stored_vecs = []
            for row in rows:
                blob = row[3]
                n = len(blob) // 4
                stored_vecs.append(np.array(struct.unpack(f"{n}f", blob), dtype=np.float32))
            q_vecs = _embed_ollama([query], self._model, self._host)
            if q_vecs is not None:
                q_vec = q_vecs[0]
                q_vec /= (np.linalg.norm(q_vec) + 1e-8)
                # Guard: if any stored blob has a different dim than the query (e.g. some rows
                # were written with the TF-IDF fallback while Ollama was down), the dot product
                # would crash — fall back to re-embedding everything with TF-IDF for consistency.
                if any(v.shape != q_vec.shape for v in stored_vecs):
                    stored_vecs = None
            else:
                # Ollama died between add and search — fall through to TF-IDF
                stored_vecs = None
        else:
            stored_vecs = None

        if stored_vecs is None:
            # TF-IDF fallback: re-embed ALL texts + query together so dimensions match.
            corpus = [r[1] for r in rows] + [query]
            all_vecs, _ = _embed_tfidf(corpus)
            stored_vecs = list(all_vecs[:-1])
            q_vec = all_vecs[-1]

        scores = [float(np.dot(q_vec, v) / (np.linalg.norm(v) + 1e-8))
                  for v in stored_vecs]
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(rows[i][1], json.loads(rows[i][2]), scores[i]) for i in top_idx]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


# ── Index events from the episodic memory store ───────────────────────────────

def index_from_memory_store(memory_dir: str, limit: int = 500) -> int:
    """Pull recent events from memory_store.jsonl and add them to the vector DB.

    Skips entries that are already in the vector DB (by checking count vs stored
    progress). Returns the number of new entries added.
    """
    import glob

    mem = SemanticMemory(memory_dir)
    already = mem.count()

    jsonl_files = glob.glob(os.path.join(memory_dir, "*.jsonl"))
    events: list[dict] = []
    for path in jsonl_files:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception:
            pass

    new_events = events[already:][:limit]
    if not new_events:
        mem.close()
        return 0

    texts, metas = [], []
    for ev in new_events:
        parts = []
        if ev.get("map"):
            parts.append(f"map {ev['map']}")
        if ev.get("terminal"):
            parts.append(f"episode ended: {ev['terminal']}")
        if ev.get("nearest_enemy"):
            parts.append(f"near {ev['nearest_enemy']}")
        if ev.get("region"):
            parts.append(f"region {ev['region']}")
        if ev.get("weapon"):
            parts.append(f"using {ev['weapon']}")
        if ev.get("health") is not None:
            parts.append(f"health {ev['health']:.0f}")
        texts.append(" ".join(parts) if parts else json.dumps(ev)[:120])
        metas.append({k: ev[k] for k in ("map", "terminal", "nearest_enemy", "region")
                      if k in ev})

    mem.add_batch(texts, metas)
    n = len(texts)
    mem.close()
    return n


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import sys

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)
    from config import Config
    cfg = Config()

    p = argparse.ArgumentParser(description="Semantic memory (vector DB) for HeLLMind.")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("recall", help="Semantic recall by free-text query")
    r.add_argument("query", nargs="+")
    r.add_argument("--top-k", type=int, default=5)

    sub.add_parser("index", help="Index episodic memory events into the vector DB")
    sub.add_parser("stats", help="Show vector DB statistics")

    args = p.parse_args()
    mem = SemanticMemory(cfg.memory_dir)

    if args.cmd == "recall":
        query = " ".join(args.query)
        results = mem.search(query, top_k=args.top_k)
        if not results:
            print("No entries in the vector DB yet — run `doom-cli semantic index` first.")
        else:
            print(f"Semantic recall: '{query}' → top {len(results)}\n")
            for text, meta, score in results:
                print(f"  [{score:.3f}] {text}")
                if meta:
                    print(f"          {meta}")

    elif args.cmd == "index":
        n = index_from_memory_store(cfg.memory_dir)
        print(f"Indexed {n} new events. Total: {mem.count()}")

    elif args.cmd == "stats":
        print(f"Vector DB: {mem._db_path}")
        print(f"  entries:       {mem.count()}")
        print(f"  embed backend: {'ollama (' + mem._model + ')' if mem._can_ollama() else 'TF-IDF (fallback)'}")

    else:
        p.print_help()

    mem.close()


if __name__ == "__main__":
    main()
