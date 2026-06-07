"""Generate a complete HTML report after a run: metrics, charts, the math/formulas behind each
number, and concrete recommendations. Self-contained (inline CSS + base64 charts) so it opens in
any browser with no server.

    from writer.html_report import write_report
    write_report(metrics_dict, "reports/run.html", meta={"map": "MAP01", "episodes": 10})

`render_html` is pure (returns a string) so it's unit-testable; charts degrade to nothing if
matplotlib is missing.
"""
import base64
import datetime as _dt
import html
import io
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- charts
def _bar_png(title: str, labels: List[str], values: List[float], color="#ff7a45") -> str:
    """A horizontal bar chart as a base64 data-URI (empty string if matplotlib is absent)."""
    if not labels:
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    fig, ax = plt.subplots(figsize=(5.2, max(1.4, 0.42 * len(labels))), dpi=110)
    ax.barh(labels[::-1], values[::-1], color=color)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _img(uri: str) -> str:
    return f'<img src="{uri}" style="max-width:100%">' if uri else ""


# --------------------------------------------------------------------------- helpers
def _pct(x) -> str:
    try:
        return f"{float(x) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _num(x, d=1) -> str:
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def _table(rows: List[tuple]) -> str:
    trs = "".join(
        f"<tr><td class='k'>{html.escape(str(k))}</td><td class='v'>{v}</td>"
        f"<td class='d'>{html.escape(str(d))}</td></tr>" for k, v, d in rows)
    return f"<table>{trs}</table>"


# --------------------------------------------------------------------------- formulas
_FORMULAS = [
    ("shooting accuracy", "hits_landed / shots_fired",
     "fraction of shots that connect — raw trigger quality"),
    ("shots per kill", "shots_fired / kills",
     "ammo discipline — lower is sharper aim"),
    ("aim offset", "mean |enemy_center_x − 0.5| · 2  (0 = dead-centre, 1 = screen edge)",
     "how well it keeps the enemy centred BEFORE firing"),
    ("wasted shots", "shots_with_no_enemy_on_screen / shots_fired",
     "spray discipline — high = firing at nothing"),
    ("kill conversion", "kills / distinct_enemies_seen",
     "does it finish what it sees?"),
    ("exit progress", "1 − closest_distance_to_exit / spawn_to_exit_distance",
     "how close to the level exit (exit read from the WAD)"),
    ("explored fraction", "visited_cells / map_cells_in_wall_bbox",
     "share of the real level area touched"),
    ("revisit rate", "(position_samples − unique_cells) / position_samples",
     "circling — high means re-treading old ground"),
    ("score (auto-loop)", "4·exit_rate + 1.5·exit_progress + 3·explored + … ",
     "the scalar the self-tuning loop maximises"),
    ("shaped reward", "base + KILL·Δkills + HIT·Δhits − DAMAGE·Δhp + ENGAGE·centred + explore_bonuses",
     "the per-step training signal (see reward breakdown)"),
]


def _recommendations(metrics: Dict) -> List[str]:
    """Concrete 'what to adjust' from the auto-loop's own diagnosis + a few rule-of-thumb reads."""
    recs: List[str] = []
    try:
        from rl.autonomous import propose
        _new, reason = propose({}, metrics)
        recs.append(f"Auto-tuner would change: {reason}")
    except Exception:
        pass
    if metrics.get("wasted_shot_rate", 0) > 0.4:
        recs.append("High wasted-shot rate → raise MISS_PENALTY / ENGAGEMENT_REWARD so it fires "
                    "only when an enemy is centred.")
    if metrics.get("aim_offset", 0) > 0.5:
        recs.append("Aim offset high → the net isn't centring enemies; boost ENGAGEMENT_REWARD "
                    "(or pre-train aim on defend_the_center).")
    rb = metrics.get("reward_breakdown", {}) or {}
    if rb.get("explore", 0) > 0.6:
        recs.append("Reward is exploration-dominated → if the goal is combat, cut COVERAGE/RND so "
                    "combat/engage isn't drowned out.")
    if metrics.get("revisit_rate", 0) > 0.85:
        recs.append("Circling (high revisit) → raise FRONTIER_REWARD / RND_SCALE (anti-circle).")
    if metrics.get("death_rate", 0) > 0.5:
        recs.append("Dies often → raise DEATH_PENALTY / DAMAGE_TAKEN_PENALTY; it sees its HEALTH.")
    if not recs:
        recs.append("No red flags — accumulate more training frames to push the weakest metric up.")
    return recs


_CSS = """
body{background:#1a1410;color:#e8dcc8;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:28px}
h1{color:#ff7a45;margin:0 0 4px} h2{color:#ff7a45;border-bottom:1px solid #534637;padding-bottom:4px;margin-top:28px}
.sub{color:#988b78;margin-bottom:18px}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.card{background:#241c16;border:1px solid #534637;border-radius:8px;padding:10px 14px;min-width:120px}
.card .big{font-size:22px;color:#ff7a45;font-weight:600} .card .lbl{color:#988b78;font-size:12px}
table{border-collapse:collapse;width:100%;margin:6px 0} td{padding:5px 8px;border-bottom:1px solid #332a22;vertical-align:top}
td.k{color:#e8dcc8;width:34%} td.v{color:#ff7a45;font-weight:600;width:16%} td.d{color:#988b78;font-size:12px}
.grid{display:flex;flex-wrap:wrap;gap:18px} .grid>div{flex:1;min-width:320px}
.rec{background:#241c16;border-left:3px solid #ff7a45;padding:8px 12px;margin:6px 0;border-radius:4px}
code{background:#241c16;color:#ffd9a8;padding:1px 5px;border-radius:3px}
.formula td.v{font-family:ui-monospace,Menlo,monospace;color:#ffd9a8;font-weight:400;width:42%}
"""


def render_html(metrics: Dict, meta: Optional[Dict] = None) -> str:
    """Full self-contained HTML report from an eval metrics snapshot."""
    m, meta = metrics or {}, meta or {}
    cov = m.get("map_coverage", {}) or {}
    when = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"HeLLMind run report — {meta.get('map', m.get('map', '?'))}"

    cards = [
        ("exit progress", _pct(m.get("exit_progress", 0))),
        ("explored", _pct(cov.get("explored_fraction", m.get("explored_fraction", 0)))),
        ("kills/ep", _num(m.get("kills_per_episode", 0))),
        ("accuracy", _pct(m.get("shooting_accuracy", 0))),
        ("aim offset", _num(m.get("aim_offset", 0), 2)),
        ("death rate", _pct(m.get("death_rate", 0))),
    ]
    cards_html = "".join(f"<div class='card'><div class='big'>{v}</div>"
                         f"<div class='lbl'>{k}</div></div>" for k, v in cards)

    aim = _table([
        ("accuracy", _pct(m.get("shooting_accuracy", 0)), "hits / shots"),
        ("shots per kill", _num(m.get("shots_per_kill", 0)), "ammo discipline (lower=better)"),
        ("aim offset", _num(m.get("aim_offset", 0), 2), "0 = enemy dead-centre"),
        ("wasted shots", _pct(m.get("wasted_shot_rate", 0)), "fired with no enemy on screen"),
        ("kill conversion", _pct(m.get("kill_conversion", 0)), "killed / seen"),
        ("nearest enemy", _num(m.get("nearest_enemy_dist", 0), 0), "avg distance (positioning)"),
    ])
    move = _table([
        ("explored", _pct(cov.get("explored_fraction", 0)), f"{int(cov.get('cells_visited',0))} cells"),
        ("revisit rate", _pct(m.get("revisit_rate", 0)), "circling"),
        ("idle/stuck", _pct(m.get("idle_rate", 0)), "steps barely moving"),
        ("frontier reach", _num(m.get("frontier_reach", 0), 0), "max distance from spawn"),
        ("distance/ep", _num(m.get("distance_per_episode", 0), 0), "units travelled"),
    ])
    surv = _table([
        ("hits taken/ep", _num(m.get("hits_taken_per_episode", 0)), "times damaged"),
        ("low-health time", _pct(m.get("low_health_fraction", 0)), "HP < 30 (where it dies)"),
        ("out of ammo", _pct(m.get("out_of_ammo_fraction", 0)), "ran dry"),
        ("heals consumed", _num(m.get("heals_consumed", 0)), "medikits/stimpacks"),
        ("decisiveness", _num(m.get("action_entropy_normalized", 0), 2), "0=fixed, 1=random"),
    ])

    rb = m.get("reward_breakdown", {}) or {}
    wu = m.get("weapons_used", {}) or {}
    dist = m.get("action_distribution", {}) or {}
    charts = "".join(_img(c) for c in [
        _bar_png("Reward breakdown (what it optimises)", list(rb.keys()),
                 [rb[k] for k in rb], "#80dc80"),
        _bar_png("Weapon usage (share of time)", list(wu.keys()), [wu[k] for k in wu], "#ffd000"),
        _bar_png("Action distribution", list(dist.keys())[:12],
                 [dist[k] for k in list(dist)[:12]], "#ff7a45"),
    ])

    formulas = "<table class='formula'>" + "".join(
        f"<tr><td class='k'>{html.escape(n)}</td><td class='v'>{html.escape(f)}</td>"
        f"<td class='d'>{html.escape(d)}</td></tr>" for n, f, d in _FORMULAS) + "</table>"
    recs = "".join(f"<div class='rec'>{html.escape(r)}</div>" for r in _recommendations(m))

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="sub">{html.escape(str(meta.get('brain','')))} · {int(m.get('episodes',0))} episodes · {when}</div>
<div class="cards">{cards_html}</div>
<h2>Charts</h2><div class="grid"><div>{charts}</div></div>
<h2>AIM</h2>{aim}
<h2>MOVEMENT</h2>{move}
<h2>SURVIVAL &amp; POLICY</h2>{surv}
<h2>Recommendations — what to adjust</h2>{recs}
<h2>Formulas &amp; math behind the numbers</h2>{formulas}
<div class="sub" style="margin-top:24px">Generated by HeLLMind · writer/html_report.py</div>
</body></html>"""


def write_report(metrics: Dict, path: str, meta: Optional[Dict] = None) -> str:
    """Render and write the report; returns the path."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(render_html(metrics, meta))
    return path
