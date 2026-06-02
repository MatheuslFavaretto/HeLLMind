"""Bestiary — a factual, persistent world model of the monsters the agent has met.

Built entirely from ViZDoom telemetry (actor names + positions + velocities), accumulated
ACROSS runs, never from a vision model. For each monster it records how often it's met,
how it attacks (ranged vs melee — derived from its projectiles and whether it charges the
player), the weapon the agent tends to use against it, how close it gets, and the episode
outcomes when it's present (a correlational threat signal). Written to Obsidian as a
`Bestiary.md` the graph links to; the raw facts live in `.memory/world/enemies.json`.
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def display_name(actor: str) -> str:
    """'DoomImp' -> 'Imp', 'ShotgunGuy' -> 'Shotgun Guy' (nicer than the class name)."""
    s = actor[4:] if actor.startswith("Doom") else actor
    out = []
    for i, c in enumerate(s):
        if c.isupper() and i and not s[i - 1].isupper():
            out.append(" ")
        out.append(c)
    return "".join(out)


def _confidence(encounters: int) -> str:
    return "high" if encounters >= 20 else "medium" if encounters >= 5 else "low"


def threat_multipliers(store: Dict[str, Any], cap: float = 3.0) -> Dict[str, float]:
    """Per-monster kill-reward multiplier learned from the bestiary: 1 + threat, where
    threat = deaths-when-present / encounters (clamped to `cap`). Deadlier monster -> the
    agent is paid more for killing it. Low-confidence (<5 encounters) monsters stay at 1.0
    so a couple of noisy episodes can't distort the reward. Empty store -> {} (all 1.0)."""
    out: Dict[str, float] = {}
    for name, s in (store or {}).items():
        enc = int(s.get("encounters", 0))
        if enc < 5:
            continue
        threat = int(s.get("outcomes", {}).get("death", 0)) / enc
        out[name] = float(min(cap, 1.0 + threat))
    return out


class BestiaryStore:
    def __init__(self, memory_dir: str) -> None:
        self.path = os.path.join(memory_dir, "world", "enemies.json")

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    _DEFAULT = {"encounters": 0, "seen": 0, "approach": 0, "killed": 0, "killed_agent": 0,
                "total": 0, "ranged": False, "dist_min": 1e9,
                "kill_weapon": {}, "outcomes": {}, "maps": {}}

    def merge(self, run: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Sum a run's per-monster accumulation into the persistent store. Per monster:
        encounters, seen, approach, killed (by agent), killed_agent (killed the agent),
        total (max on a map), ranged, dist_min, kill_weapon{slot:n}, outcomes{}, maps{}."""
        store = self.load()
        for name, r in run.items():
            s = store.setdefault(name, {k: (dict(v) if isinstance(v, dict) else v)
                                        for k, v in self._DEFAULT.items()})
            for k in self._DEFAULT:  # backfill keys added in newer versions
                s.setdefault(k, dict(self._DEFAULT[k]) if isinstance(self._DEFAULT[k], dict)
                             else self._DEFAULT[k])
            for k in ("encounters", "seen", "approach", "killed", "killed_agent"):
                s[k] += int(r.get(k, 0))
            s["total"] = max(int(s["total"]), int(r.get("total", 0)))
            s["ranged"] = bool(s["ranged"] or r.get("ranged", False))
            s["dist_min"] = min(float(s["dist_min"]), float(r.get("dist_min", 1e9)))
            for d in ("kill_weapon", "outcomes", "maps"):
                for k, v in (r.get(d, {}) or {}).items():
                    s[d][str(k)] = s[d].get(str(k), 0) + int(v)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(store, f)
        os.replace(tmp, self.path)
        return store


def _attack_style(name: str, s: Dict[str, Any]) -> str:
    from doom.entities import HITSCAN
    seen = max(1, int(s.get("seen", 0)))
    approach = int(s.get("approach", 0)) / seen
    if s.get("ranged"):
        return f"ranged (throws projectiles); closes in {approach:.0%} of the time"
    if name in HITSCAN:
        return f"ranged (hitscan — fires bullets); closes in {approach:.0%} of the time"
    return ("melee — charges the player" if approach >= 0.5
            else "melee — holds ground / circles")


def write_bestiary(cfg, store: Dict[str, Any] = None) -> str:
    """Render the Bestiary note from the persisted world memory. Returns its path (or '')."""
    store = store if store is not None else BestiaryStore(cfg.memory_dir).load()
    if not store:
        return ""
    # Most-encountered first (the agent's real experience).
    order = sorted(store.items(), key=lambda kv: -int(kv[1].get("encounters", 0)))
    lines = [
        "---", "type: bestiary",
        f"updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "tags: [bestiary, world-model, doom-rl]", "---", "",
        "# Bestiary — monsters the agent has met",
        "",
        "Facts from ViZDoom telemetry across runs (actor names, positions, velocities). "
        "Threat = share of episodes that ended in death while this monster was present "
        "(correlational, not proof of the killer).",
        "",
    ]
    # Bar chart (kills by the agent vs deaths caused, per monster) -> attachments.
    chart_md = ""
    try:
        from writer.bestiary_chart import render_bestiary_chart
        out_dir_a = os.path.join(cfg.vault_path, getattr(cfg, "dir_attachments", "attachments"))
        if render_bestiary_chart(store, os.path.join(out_dir_a, "bestiary.png")):
            chart_md = "![[bestiary.png]]\n\n"
    except Exception:
        chart_md = ""
    if chart_md:
        lines.append(chart_md)

    for name, s in order:
        enc = int(s.get("encounters", 0))
        deaths = int(s.get("outcomes", {}).get("death", 0))
        threat = deaths / enc if enc else 0.0
        kw = s.get("kill_weapon", {})
        best_w = max(kw.items(), key=lambda kv: kv[1])[0] if kw else "?"
        maps = ", ".join(sorted(s.get("maps", {}))) or "?"
        lines += [
            f"## {display_name(name)}  ·  [[Bestiary]]",
            f"- **Seen:** {int(s.get('total', 0))} on the map · met in {enc} episode(s) "
            f"_(confidence: {_confidence(enc)})_",
            f"- **Killed by the agent:** {int(s.get('killed', 0))}",
            f"- **Killed the agent:** {int(s.get('killed_agent', 0))} time(s)",
            f"- **Attack style:** {_attack_style(name, s)}",
            f"- **Threat (death-rate when present):** {threat:.0%}",
            f"- **Best weapon vs it (most kills):** slot {best_w}",
            f"- **Closest it got:** {int(float(s.get('dist_min', 0)))} map units",
            f"- **Seen on:** {maps}",
            "",
        ]
    out_dir = os.path.join(cfg.vault_path, "70-bestiary")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Bestiary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main() -> None:
    from config import Config
    cfg = Config()
    p = write_bestiary(cfg)
    print(f"[bestiary] wrote {p}" if p else "[bestiary] no monster data yet — train in "
          "campaign mode first.")


if __name__ == "__main__":
    main()
