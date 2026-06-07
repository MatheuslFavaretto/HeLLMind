"""Export HeLLMind metrics to Prometheus (push-gateway or a textfile) for time-series dashboards.

Eval/train produce a flat-ish metrics dict (the panels: aim_offset, wasted_shot_rate, explored,
reward_breakdown, weapons_used, ...). This flattens it into Prometheus gauges so you can scrape it
into Grafana and WATCH the agent improve across runs (aim_offset falling, kill_conversion rising).

Opt-in, env-driven (no overhead unless configured):
  PROMETHEUS_GATEWAY=localhost:9091   # push to a Prometheus Pushgateway
  PROMETHEUS_TEXTFILE=/path/run.prom  # or write a node_exporter textfile-collector file

The flattening is pure + unit-tested; the actual push needs `prometheus_client` (optional dep).
"""
import os
from typing import Dict, List, Optional, Tuple

# Keys whose values are huge arrays / not metrics — never export these.
_SKIP = {"path_cells", "path_polyline", "map_walls", "action_distribution"}


def flatten_metrics(metrics: dict, prefix: str = "hellmind") -> List[Tuple[str, Dict[str, str], float]]:
    """Flatten a metrics dict to (gauge_name, labels, value) triples.

    Scalars → `prefix_key`. Dict-of-scalars (reward_breakdown, weapons_used, accuracy_by_weapon,
    map_coverage…) → `prefix_key{item="sub"}`. Non-numeric / array values are skipped."""
    out: List[Tuple[str, Dict[str, str], float]] = []
    for key, val in (metrics or {}).items():
        if key in _SKIP:
            continue
        name = f"{prefix}_{key}"
        if isinstance(val, bool):
            out.append((name, {}, float(val)))
        elif isinstance(val, (int, float)):
            out.append((name, {}, float(val)))
        elif isinstance(val, dict):
            for sub, v in val.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append((name, {"item": str(sub)}, float(v)))
    return out


def export_metrics(metrics: dict, job: str = "hellmind_eval",
                   gateway: Optional[str] = None, textfile: Optional[str] = None) -> bool:
    """Push the flattened metrics to a Pushgateway and/or write a textfile. Reads
    PROMETHEUS_GATEWAY / PROMETHEUS_TEXTFILE from the env when args are omitted. Returns True if
    anything was exported. Degrades gracefully (warns) if prometheus_client isn't installed."""
    gateway = gateway or os.getenv("PROMETHEUS_GATEWAY") or None
    textfile = textfile or os.getenv("PROMETHEUS_TEXTFILE") or None
    if not gateway and not textfile:
        return False
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, write_to_textfile
    except ImportError:
        print("[prometheus] pip install prometheus_client to export metrics; skipping.")
        return False
    reg = CollectorRegistry()
    gauges: Dict[str, "Gauge"] = {}
    for name, labels, value in flatten_metrics(metrics):
        if name not in gauges:
            gauges[name] = Gauge(name, f"HeLLMind metric {name}", list(labels.keys()),
                                 registry=reg)
        (gauges[name].labels(**labels) if labels else gauges[name]).set(value)
    if textfile:
        write_to_textfile(textfile, reg)
        print(f"[prometheus] wrote {textfile}")
    if gateway:
        push_to_gateway(gateway, job=job, registry=reg)
        print(f"[prometheus] pushed to gateway {gateway} (job={job})")
    return True
