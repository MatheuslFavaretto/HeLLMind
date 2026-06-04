"""Retrieval API — "what have I learned about X?" across prior runs.

Phase 1: keyword / filter search over the SQLite store.
Phase 2 upgrade: swap the search body for cosine similarity over Ollama embeddings
(nomic-embed-text) — the API stays identical.

    from writer.recall import recall, recall_map, recall_deaths
    recall("deaths on MAP02")          # top-k lessons + events mentioning MAP02
    recall_map("MAP01")                # all events on MAP01
    recall_deaths(max_hp=30)           # deaths at low HP
"""
from typing import Any, Dict, List, Optional

from writer import db as _db

_MAP_TAGS = [f"MAP{i:02d}" for i in range(1, 33)]
_EVENT_TYPES = ("death", "success", "exit", "timeout")


def recall(
    query: str,
    memory_dir: str = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Return top-k relevant lessons + events matching the query string.

    Heuristically extracts a map name and/or event type from the query to
    filter events, then also searches lessons by keyword. Results are sorted
    by timestamp (most recent first) and capped to top_k.
    """
    if memory_dir is None:
        from config import Config
        memory_dir = Config().memory_dir

    ql = query.lower()

    event_type: Optional[str] = next((t for t in _EVENT_TYPES if t in ql), None)
    map_name: Optional[str] = next((m for m in _MAP_TAGS if m.lower() in ql), None)

    results: List[Dict[str, Any]] = []

    for l in _db.query_lessons(memory_dir, keyword=query, limit=top_k):
        results.append({
            "source": "lesson",
            "ts": l.get("ts", ""),
            "title": l.get("title", ""),
            "body": l.get("insight", ""),
            "evidence": l.get("evidence", ""),
        })

    for e in _db.query_events(memory_dir, event_type=event_type,
                               map_name=map_name, limit=top_k):
        enemy = e.get("nearest_enemy") or ""
        region = e.get("region") or ""
        weapon = e.get("weapon")
        ctx = "  ".join(filter(None, [
            f"region={region}" if region else "",
            f"weapon={weapon}" if weapon is not None else "",
            f"near={enemy}" if enemy else "",
        ]))
        results.append({
            "source": "event",
            "ts": e.get("ts", ""),
            "title": f"{e.get('type', '?')} on {e.get('map', '?')}",
            "body": (
                f"kills={e.get('kills')}  coverage={e.get('coverage')}  "
                f"health={e.get('health')}  length={e.get('length')}"
                + (f"  {ctx}" if ctx else "")
            ),
            "evidence": f"run={e.get('run', '')}",
        })

    results.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return results[:top_k]


def recall_map(map_name: str, memory_dir: str = None) -> List[Dict[str, Any]]:
    """All stored events for a specific map (most recent first)."""
    if memory_dir is None:
        from config import Config
        memory_dir = Config().memory_dir
    return _db.query_events(memory_dir, map_name=map_name, limit=500)


def recall_deaths(
    memory_dir: str = None,
    min_hp: float = None,
    max_hp: float = None,
) -> List[Dict[str, Any]]:
    """Deaths filtered by health at the moment of death."""
    if memory_dir is None:
        from config import Config
        memory_dir = Config().memory_dir
    events = _db.query_events(memory_dir, event_type="death", limit=1000)
    if min_hp is not None:
        events = [e for e in events
                  if e.get("health") is not None and e["health"] >= min_hp]
    if max_hp is not None:
        events = [e for e in events
                  if e.get("health") is not None and e["health"] <= max_hp]
    return events


def recall_enemy(
    enemy_name: str,
    memory_dir: str = None,
) -> List[Dict[str, Any]]:
    """Episodes where nearest_enemy matched (partial, case-insensitive)."""
    if memory_dir is None:
        from config import Config
        memory_dir = Config().memory_dir
    con = _db.connect(memory_dir)
    rows = con.execute(
        "SELECT * FROM events WHERE nearest_enemy LIKE ? ORDER BY ts DESC LIMIT 200",
        (f"%{enemy_name}%",),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def recall_region(
    region: str,
    memory_dir: str = None,
) -> List[Dict[str, Any]]:
    """Episodes that ended in a specific map region (e.g. '1x2')."""
    if memory_dir is None:
        from config import Config
        memory_dir = Config().memory_dir
    con = _db.connect(memory_dir)
    rows = con.execute(
        "SELECT * FROM events WHERE region = ? ORDER BY ts DESC LIMIT 200",
        (region,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def recall_hypotheses(memory_dir: str = None, status: str = None) -> List[Dict[str, Any]]:
    """Open (or all) hypotheses from the experiment ledger."""
    if memory_dir is None:
        from config import Config
        memory_dir = Config().memory_dir
    return _db.query_hypotheses(memory_dir, status=status)


def recall_experiments(memory_dir: str = None) -> List[Dict[str, Any]]:
    """All recorded experiments, most recent first."""
    if memory_dir is None:
        from config import Config
        memory_dir = Config().memory_dir
    return _db.query_experiments(memory_dir)
