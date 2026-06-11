"""Thesis-grade experiment report — every number with its uncertainty and methodology.

What "thesis-grade" means here, concretely:
  • every per-episode metric reported as mean ± 95% CI (Student's t) with n stated;
  • the full optimisation TRAJECTORY plotted, with regime boundaries marked — scores
    from different maps or metric definitions are never visually conflated;
  • the exact formulas used (composite score, route_progress, shaping, CI) printed in
    the report itself, so a reader can recompute any number;
  • an explicit limitations section (single-seed, metric eras, small n) — the report
    states what it CANNOT claim.

Motivating incident: an earlier report confidently showed "route_progress 93%" while
the agent had learned to jump into a pit (reward hacking). A report without intervals,
methodology and regime context is an anecdote generator.

    doom-cli report            → reports/thesis_report.html
"""
from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

# ---------------------------------------------------------------- chart helpers

_COL = {"keep": "#2da44e", "revert": "#cf222e", "line": "#ff7a45",
        "regime": "#8250df", "grid": "#d0d7de"}


def _fig_to_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _regime_boundaries(rows: List[dict]) -> List[int]:
    """Iterations where the comparison regime changed: an escape fired OR the map
    changed. Scores across a boundary are NOT comparable (different map, different
    config era, or different metric definition)."""
    cuts = []
    prev_map = None
    for i, r in enumerate(rows):
        cur_map = (r.get("env") or {}).get("MAPS")
        if (r.get("plateau_level") or 0) > 0:
            cuts.append(i)
        elif prev_map is not None and cur_map != prev_map:
            cuts.append(i)
        prev_map = cur_map
    return cuts


def _trajectory_chart(rows: List[dict]) -> str:
    """Score per iteration: KEEP green / REVERT red, regime boundaries as dashed
    verticals. The single most informative picture of the whole optimisation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    its = [r["iter"] for r in rows]
    scores = [r["score"] for r in rows]
    kept = [bool(r.get("kept")) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(its, scores, color=_COL["line"], lw=1, alpha=0.6, zorder=1)
    ax.scatter([i for i, k in zip(its, kept) if k],
               [s for s, k in zip(scores, kept) if k],
               color=_COL["keep"], s=18, label="KEEP", zorder=2)
    ax.scatter([i for i, k in zip(its, kept) if not k],
               [s for s, k in zip(scores, kept) if not k],
               color=_COL["revert"], s=18, label="REVERT", zorder=2)
    for c in _regime_boundaries(rows):
        ax.axvline(rows[c]["iter"], color=_COL["regime"], ls="--", lw=0.9, alpha=0.7)
    ax.set_xlabel("iteration"); ax.set_ylabel("composite score")
    ax.set_title("Optimisation trajectory — dashed verticals are regime boundaries "
                 "(scores across them are NOT comparable)")
    ax.grid(color=_COL["grid"], lw=0.4); ax.legend(loc="upper left", fontsize=8)
    return _fig_to_uri(fig)


def _metric_chart(rows: List[dict], keys: List[str], title: str,
                  ylabel: str, pct: bool = False) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    fig, ax = plt.subplots(figsize=(10, 3))
    plotted = False
    for key in keys:
        xs, ys = [], []
        for r in rows:
            v = (r.get("metrics") or {}).get(key)
            if v is not None:
                xs.append(r["iter"]); ys.append(v * (100 if pct else 1))
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, lw=1, label=key)
            plotted = True
    if not plotted:
        plt.close(fig)
        return ""
    for c in _regime_boundaries(rows):
        ax.axvline(rows[c]["iter"], color=_COL["regime"], ls="--", lw=0.9, alpha=0.5)
    ax.set_xlabel("iteration"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(color=_COL["grid"], lw=0.4); ax.legend(fontsize=8)
    return _fig_to_uri(fig)


# ---------------------------------------------------------------- report body

def _ci_rows(stats: Dict[str, dict]) -> str:
    """episode_stats → an HTML table body: mean ± CI (n, std, median)."""
    out = []
    for name, st in sorted(stats.items()):
        half = (st["ci95_hi"] - st["ci95_lo"]) / 2
        out.append(
            f"<tr><td>{name}</td>"
            f"<td><b>{st['mean']:.3f} ± {half:.3f}</b></td>"
            f"<td>{st['std']:.3f}</td><td>{st['median']:.3f}</td>"
            f"<td>{st['n']}</td></tr>")
    return "\n".join(out)


_FORMULAS = """
<h2>Formulas (recompute any number)</h2>
<pre>
Composite score (SCORE_PROFILE weights w):
  score = w_acc·accuracy + w_kc·kill_conversion + w_k·min(kills,5)/5
        + w_ex·explored + w_rp·route_progress + w_er·exit_rate
        + w_w·wasted_shot_rate + w_ao·aim_offset + w_d·death_rate
  combat: w = (2.5, 1.5, 0.5, 1.0, 1.0, 2.0, −1.5, −1.0, −0.5)
  exit:   w = (1.0, 0.5, 0.25, 1.5, 2.0, 5.0, −0.75, −0.5, −1.5)

Geodesic route metric (BFS over WAD walls, 64u grid, directional steps ≤24u,
sub-32u slits passable, off-route = max_field + euclidean):
  route_progress      = 1 − d_geo(closest point reached) / d_geo(spawn)
  route_progress_best = max over episodes
  KEEP rule           = score ≥ best_regime − 0.05

Exit-proximity shaping (signed, potential-based — telescopes, unfarmable):
  r_t += EXIT_PROX_SCALE · (d_geo(s_{t−1}) − d_geo(s_t)) · 0.001

95% confidence interval (Student's t, two-sided, df = n−1):
  CI = mean ± t_{0.975,n−1} · s / √n      (s = sample std, ddof = 1)
</pre>"""

_LIMITS = """
<h2>Limitations (what this report cannot claim)</h2>
<ul>
<li><b>Single seed:</b> trajectory results come from one training run; per-eval CIs
capture episode variance, not run-to-run variance. Multi-seed replication is required
before any cross-configuration claim.</li>
<li><b>Metric eras:</b> route_progress changed definition during the project
(euclidean → geodesic → pit-exclusion). Regime boundaries (dashed lines) mark where
numbers stop being comparable; this report never aggregates across them.</li>
<li><b>Small n per evaluation:</b> 10–20 episodes per point; CIs are wide. A KEEP and
a REVERT whose intervals overlap may be noise (the loop's 0.05 margin is a heuristic,
not a significance test).</li>
<li><b>Reward hacking risk is permanent:</b> a previous metric reported 93% while
measuring an exploit. Any suspiciously pinned or non-grid-aligned value should be
investigated before being believed.</li>
</ul>"""


def render(memory_dir: str, last_n: int = 60) -> str:
    """Build the full HTML from the autonomy trail's last `last_n` iterations."""
    trail = os.path.join(memory_dir, "autonomy.jsonl")
    rows: List[dict] = []
    if os.path.exists(trail):
        with open(trail, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    rows = rows[-last_n:]
    last = rows[-1] if rows else {}
    lm = last.get("metrics") or {}
    meta = lm.get("eval_meta") or {}
    stats = lm.get("episode_stats") or {}

    # Methodology block — every condition under which the numbers were produced.
    def _row(k, v):
        return f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
    assists = meta.get("assists") or {}
    method = "\n".join([
        _row("generated (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        _row("iterations analysed", f"{len(rows)} (iters {rows[0]['iter']}–{last['iter']})"
             if rows else "0"),
        _row("episodes per evaluation", meta.get("episodes", "unrecorded (pre-meta era)")),
        _row("policy sampling", f"temperature={meta.get('temperature')}"
             if meta.get("temperature") is not None else "unrecorded"),
        _row("rng seed", meta.get("seed", "UNSEEDED — episode sampling noise present")),
        _row("map", meta.get("map", (last.get("env") or {}).get("MAPS", "?"))),
        _row("brain checkpoint", meta.get("brain", "unrecorded")),
        _row("assists", ", ".join(f"{k}={'on' if v else 'off'}"
                                  for k, v in assists.items()) or "unrecorded"),
        _row("regime boundaries in window", len(_regime_boundaries(rows))),
    ])

    charts = "".join(
        f'<img src="{uri}" style="max-width:100%"/><br/>' for uri in [
            _trajectory_chart(rows),
            _metric_chart(rows, ["route_progress", "route_progress_best"],
                          "Geodesic route penetration (honest metric era only)",
                          "% of true route", pct=True),
            _metric_chart(rows, ["explored_fraction", "exit_rate"],
                          "Exploration and exit rate", "fraction (%)", pct=True),
            _metric_chart(rows, ["kills_per_episode"], "Combat", "kills / episode"),
            _metric_chart(rows, ["death_rate", "timeout_rate"],
                          "Episode terminations", "% of episodes", pct=True),
        ] if uri)

    ci_section = (
        f"<h2>Latest evaluation — distributions (mean ± 95% CI)</h2>"
        f"<table><tr><th>metric</th><th>mean ± 95% CI</th><th>std</th>"
        f"<th>median</th><th>n</th></tr>{_ci_rows(stats)}</table>"
        if stats else
        "<h2>Latest evaluation — distributions</h2><p><i>episode_stats not present in "
        "the latest trail entry — re-run an evaluation with the current code to "
        "populate per-episode distributions.</i></p>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>HeLLMind — experiment report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:980px;
      color:#1f2328;line-height:1.45}}
 table{{border-collapse:collapse;margin:0.6rem 0}}
 td,th{{border:1px solid #d0d7de;padding:4px 10px;font-size:14px;text-align:left}}
 th{{background:#f6f8fa}}
 pre{{background:#f6f8fa;padding:12px;border-radius:6px;font-size:13px;overflow-x:auto}}
 h1{{border-bottom:2px solid #ff7a45;padding-bottom:6px}}
</style></head><body>
<h1>HeLLMind — experiment report</h1>
<p>Self-improving RL agent, solo (skill assists off). Every number below is
reproducible from <code>autonomy.jsonl</code> + the formulas section.</p>
<h2>Methodology</h2><table>{method}</table>
{ci_section}
<h2>Trajectories</h2>{charts or "<p><i>matplotlib unavailable — charts skipped.</i></p>"}
{_FORMULAS}
{_LIMITS}
</body></html>"""


def write(memory_dir: str, path: str = "reports/thesis_report.html",
          last_n: int = 60) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(memory_dir, last_n=last_n))
    return path
