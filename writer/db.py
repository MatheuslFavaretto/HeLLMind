"""SQLite cognitive memory (Phase 1 — queryable view over the append-only JSONL stores).

The write path stays JSONL (MemoryStore / CoverageStore / BestiaryStore — append-only,
safe from concurrent writers). SQLite is the READ / QUERY layer: rebuilt offline from the
JSONL/JSON files, never written to during training.

Layout:
    <memory_dir>/hellmind.db

Tables: runs, events, lessons, maps, hypotheses, experiments

    python -m writer.db build           # (re)build from JSONL stores
    python -m writer.db query MAP02     # events mentioning MAP02
    python -m writer.db query --lessons low HP
"""
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    name        TEXT PRIMARY KEY,
    ts          TEXT,
    total_steps INTEGER,
    maps        TEXT,
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run           TEXT,
    ts            TEXT,
    type          TEXT,
    map           TEXT,
    health        REAL,
    ammo          REAL,
    kills         INTEGER,
    coverage      REAL,
    length        INTEGER,
    weapon        INTEGER,
    region        TEXT,
    nearest_enemy TEXT,
    extra_json    TEXT
);

CREATE TABLE IF NOT EXISTS lessons (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT,
    run     TEXT,
    title   TEXT,
    insight TEXT,
    evidence TEXT
);

CREATE TABLE IF NOT EXISTS maps (
    map        TEXT PRIMARY KEY,
    runs       INTEGER,
    cell       REAL,
    updated    TEXT,
    cells_json TEXT,
    walls_json TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT,
    title      TEXT,
    body       TEXT,
    metric     TEXT,
    direction  TEXT,
    confidence REAL,
    status     TEXT DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS experiments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT,
    hypothesis_id INTEGER,
    param         TEXT,
    old_val       TEXT,
    new_val       TEXT,
    result        TEXT,
    confidence    REAL,
    notes         TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_map  ON events(map);
CREATE INDEX IF NOT EXISTS idx_events_run  ON events(run);
"""


def _db_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "hellmind.db")


def connect(memory_dir: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and apply the schema."""
    os.makedirs(memory_dir, exist_ok=True)
    con = sqlite3.connect(_db_path(memory_dir))
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    # Forward-compatible migrations: add new columns to existing DBs.
    for col, typedef in [
        ("weapon", "INTEGER"),
        ("region", "TEXT"),
        ("nearest_enemy", "TEXT"),
    ]:
        try:
            con.execute(f"ALTER TABLE events ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # column already exists
    con.commit()
    return con


# ---------------------------------------------------------------------------
# Build  (JSONL / JSON stores → SQLite)
# ---------------------------------------------------------------------------

def build(memory_dir: str) -> int:
    """Rebuild the SQLite DB from the JSONL/JSON stores. Returns total rows loaded."""
    from writer.memory_store import MemoryStore

    con = connect(memory_dir)
    total = 0

    # Events (episodic/events.jsonl) — full reload
    events = MemoryStore.read_events(memory_dir)
    con.execute("DELETE FROM events")
    _KNOWN = {"run", "ts", "type", "map", "health", "ammo", "kills",
              "coverage", "length", "weapon", "region", "nearest_enemy"}
    for e in events:
        extra = {k: v for k, v in e.items() if k not in _KNOWN}
        con.execute(
            "INSERT INTO events"
            " (run, ts, type, map, health, ammo, kills, coverage, length,"
            "  weapon, region, nearest_enemy, extra_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                e.get("run", ""),
                e.get("ts", ""),
                e.get("type", ""),
                e.get("map", ""),
                _to_float(e.get("health")),
                _to_float(e.get("ammo")),
                _to_int(e.get("kills")),
                _to_float(e.get("coverage")),
                _to_int(e.get("length")),
                _to_int(e.get("weapon")),
                e.get("region", ""),
                e.get("nearest_enemy", ""),
                json.dumps(extra) if extra else None,
            ),
        )
        total += 1

    # Lessons (lessons/lessons.jsonl) — full reload
    lessons_path = os.path.join(memory_dir, "lessons", "lessons.jsonl")
    if os.path.exists(lessons_path):
        con.execute("DELETE FROM lessons")
        with open(lessons_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    l = json.loads(line)
                except json.JSONDecodeError:
                    continue
                con.execute(
                    "INSERT INTO lessons (ts, run, title, insight, evidence)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (l.get("ts", ""), l.get("run", ""), l.get("title", ""),
                     l.get("insight", ""), l.get("evidence", "")),
                )
                total += 1

    # Runs (autonomy.jsonl) — one row per auto-loop iteration, full reload.
    # The write-path is autonomy.jsonl (the resumable loop trail); this mirrors it
    # into the queryable `runs` table so `db query` / experiment.py can join on it.
    autonomy_path = os.path.join(memory_dir, "autonomy.jsonl")
    if os.path.exists(autonomy_path):
        con.execute("DELETE FROM runs")
        with open(autonomy_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                env = rec.get("env", {}) or {}
                metrics = rec.get("metrics", {}) or {}
                name = f"iter-{_to_int(rec.get('iter')) or 0:03d}"
                con.execute(
                    "INSERT OR REPLACE INTO runs"
                    " (name, ts, total_steps, maps, config_json)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        name,
                        rec.get("ts", ""),
                        _to_int(metrics.get("mean_episode_length")),
                        env.get("MAPS", ""),
                        json.dumps({"env": env, "metrics": metrics,
                                    "score": rec.get("score"),
                                    "kept": rec.get("kept"),
                                    "reason": rec.get("reason", "")}),
                    ),
                )
                total += 1

    # Maps (coverage/*.json) — full reload
    cov_dir = os.path.join(memory_dir, "coverage")
    if os.path.isdir(cov_dir):
        con.execute("DELETE FROM maps")
        for fname in os.listdir(cov_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(cov_dir, fname), encoding="utf-8") as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            con.execute(
                "INSERT OR REPLACE INTO maps"
                " (map, runs, cell, updated, cells_json, walls_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rec.get("map", fname.replace(".json", "")),
                    _to_int(rec.get("runs")),
                    _to_float(rec.get("cell")),
                    rec.get("updated", ""),
                    json.dumps(rec.get("cells", {})),
                    json.dumps(rec.get("walls", [])),
                ),
            )
            total += 1

    con.commit()
    con.close()
    return total


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query_events(
    memory_dir: str,
    event_type: str = None,
    map_name: str = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    con = connect(memory_dir)
    clauses, params = [], []
    if event_type:
        clauses.append("type = ?")
        params.append(event_type)
    if map_name:
        clauses.append("map = ?")
        params.append(map_name)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = con.execute(
        f"SELECT * FROM events {where} ORDER BY ts DESC LIMIT ?",
        params + [limit],
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def query_lessons(
    memory_dir: str,
    keyword: str = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    con = connect(memory_dir)
    if keyword:
        pat = f"%{keyword}%"
        rows = con.execute(
            "SELECT * FROM lessons"
            " WHERE title LIKE ? OR insight LIKE ? OR evidence LIKE ?"
            " ORDER BY ts DESC LIMIT ?",
            (pat, pat, pat, limit),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM lessons ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def query_runs(memory_dir: str, limit: int = 50) -> List[Dict[str, Any]]:
    con = connect(memory_dir)
    rows = con.execute(
        "SELECT name, ts, total_steps, maps, config_json FROM runs"
        " ORDER BY name DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def query_maps(memory_dir: str) -> List[Dict[str, Any]]:
    con = connect(memory_dir)
    rows = con.execute("SELECT map, runs, cell, updated FROM maps ORDER BY map").fetchall()
    con.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Write helpers (Phases 2 & 3 — hypotheses / experiments)
# ---------------------------------------------------------------------------

def insert_hypothesis(
    memory_dir: str,
    title: str,
    body: str,
    metric: str,
    direction: str,
    confidence: float,
) -> int:
    """Insert a new hypothesis. Returns its row id."""
    from datetime import datetime, timezone
    con = connect(memory_dir)
    cur = con.execute(
        "INSERT INTO hypotheses (ts, title, body, metric, direction, confidence, status)"
        " VALUES (?, ?, ?, ?, ?, ?, 'open')",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            title, body, metric, direction, float(confidence),
        ),
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def update_hypothesis_status(memory_dir: str, hypothesis_id: int, status: str) -> None:
    """Set status to 'confirmed' or 'rejected'."""
    con = connect(memory_dir)
    con.execute(
        "UPDATE hypotheses SET status = ? WHERE id = ?", (status, hypothesis_id)
    )
    con.commit()
    con.close()


def insert_experiment(
    memory_dir: str,
    param: str,
    old_val: str,
    new_val: str,
    result: str,
    confidence: float,
    hypothesis_id: Optional[int] = None,
    notes: str = "",
) -> int:
    """Record an experiment result. Returns its row id."""
    from datetime import datetime, timezone
    con = connect(memory_dir)
    cur = con.execute(
        "INSERT INTO experiments"
        " (ts, hypothesis_id, param, old_val, new_val, result, confidence, notes)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            hypothesis_id,
            param, old_val, new_val, result,
            float(confidence), notes,
        ),
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def query_experiments(memory_dir: str, limit: int = 50) -> List[Dict[str, Any]]:
    con = connect(memory_dir)
    rows = con.execute(
        "SELECT * FROM experiments ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def query_hypotheses(memory_dir: str, status: str = None) -> List[Dict[str, Any]]:
    con = connect(memory_dir)
    if status:
        rows = con.execute(
            "SELECT * FROM hypotheses WHERE status = ? ORDER BY ts DESC", (status,)
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM hypotheses ORDER BY ts DESC"
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------

def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="HeLLMind SQLite cognitive memory.")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("build", help="Rebuild hellmind.db from JSONL/JSON stores.")

    q = sub.add_parser("query", help="Query events or lessons.")
    q.add_argument("keyword", nargs="?", default=None, help="Keyword to search.")
    q.add_argument("--type", dest="event_type", default=None,
                   help="Filter events by type (death/success/exit/timeout).")
    q.add_argument("--map", dest="map_name", default=None,
                   help="Filter events by map name (e.g. MAP01).")
    q.add_argument("--lessons", action="store_true", help="Search lessons instead of events.")
    q.add_argument("--hypotheses", action="store_true", help="List hypotheses.")
    q.add_argument("--experiments", action="store_true", help="List experiments.")
    q.add_argument("--runs", action="store_true", help="List auto-loop runs.")

    args = p.parse_args()
    from config import Config
    cfg = Config()

    if args.cmd == "build":
        n = build(cfg.memory_dir)
        print(f"[db] built hellmind.db — {n} rows loaded into {_db_path(cfg.memory_dir)}")

    elif args.cmd == "query":
        if args.runs:
            for r in query_runs(cfg.memory_dir):
                cfgj = json.loads(r["config_json"] or "{}")
                m = cfgj.get("metrics", {})
                print(f"{r['name']}  map={r['maps']:6} score={cfgj.get('score')}  "
                      f"explored={m.get('explored_fraction')} kills={m.get('kills_per_episode')} "
                      f"kept={cfgj.get('kept')}")
        elif args.hypotheses:
            for r in query_hypotheses(cfg.memory_dir):
                print(f"[{r['status']:8}] {r['title']}  (metric={r['metric']}, "
                      f"conf={r['confidence']:.2f})")
        elif args.experiments:
            for r in query_experiments(cfg.memory_dir):
                print(f"[{r['result']:10}] {r['param']}: {r['old_val']}→{r['new_val']}  "
                      f"conf={r['confidence']:.2f}")
        elif args.lessons:
            for r in query_lessons(cfg.memory_dir, keyword=args.keyword):
                print(f"[{r['ts']}] {r['title']}")
                print(f"  {r['insight']}")
        else:
            rows = query_events(
                cfg.memory_dir,
                event_type=args.event_type,
                map_name=args.map_name or args.keyword,
            )
            for r in rows:
                print(f"[{r['ts']}] {r['type']:8} map={r['map']:6} "
                      f"kills={r['kills']} cov={r['coverage']} hp={r['health']}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
