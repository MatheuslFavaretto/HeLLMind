"""Introspection — prove there's a real neural network, and summarise the intelligence.

`doom-cli intel` calls this to show: the policy network's architecture (layers, depth,
parameter count, input/output shapes — hard proof it's a real CNN, not a lookup table),
how much it has trained, the cognitive memory it has accumulated, the best run so far, and
disk usage. Everything reads files on disk; nothing boots ViZDoom.
"""
import glob
import json
import os
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Neural network introspection
# ---------------------------------------------------------------------------

def brain_report(path: str) -> Dict:
    """Inspect a saved PPO/RecurrentPPO brain (.zip) WITHOUT an env. Returns architecture
    facts: parameter count, per-layer breakdown, depth, obs/action shapes."""
    if not path or not os.path.exists(path):
        return {"exists": False}
    from stable_baselines3 import PPO
    try:
        model = PPO.load(path, device="cpu")
    except Exception:
        # LSTM brains need sb3-contrib; try that before giving up.
        try:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(path, device="cpu")
        except Exception as e:
            return {"exists": True, "error": f"{type(e).__name__}: {e}", "path": path}

    policy = model.policy
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    # Per-layer breakdown of the layers that actually hold weights (the real "depth").
    layers: List[Dict] = []
    for name, mod in policy.named_modules():
        kind = mod.__class__.__name__
        if kind in ("Conv2d", "Linear"):
            p = sum(x.numel() for x in mod.parameters())
            desc = _describe_layer(mod, kind)
            layers.append({"name": name, "kind": kind, "params": p, "desc": desc})

    obs_shape = tuple(getattr(model.observation_space, "shape", ()) or ())
    n_actions = int(getattr(model.action_space, "n", 0) or 0)
    return {
        "exists": True,
        "path": path,
        "size_mb": round(os.path.getsize(path) / 1e6, 1),
        "policy_class": policy.__class__.__name__,
        "total_params": total,
        "trainable_params": trainable,
        "weight_layers": layers,
        "depth": len(layers),                 # layers that carry weights
        "obs_shape": obs_shape,
        "n_actions": n_actions,
        "device": str(policy.device),
    }


def _describe_layer(mod, kind: str) -> str:
    import torch.nn as nn
    if isinstance(mod, nn.Conv2d):
        return (f"Conv2d({mod.in_channels}→{mod.out_channels}, "
                f"k={mod.kernel_size[0]}, s={mod.stride[0]})")
    if isinstance(mod, nn.Linear):
        return f"Linear({mod.in_features}→{mod.out_features})"
    return kind


# ---------------------------------------------------------------------------
# Training / memory / disk stats
# ---------------------------------------------------------------------------

def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def training_stats(checkpoint_dir: str, name_prefix: str) -> Dict:
    """Steps trained (from the highest checkpoint number) + checkpoint count."""
    steps_files = glob.glob(os.path.join(checkpoint_dir, f"{name_prefix}_[0-9]*_steps.zip"))
    best_steps = 0
    for f in steps_files:
        base = os.path.basename(f)
        try:
            n = int(base.replace(f"{name_prefix}_", "").replace("_steps.zip", ""))
            best_steps = max(best_steps, n)
        except ValueError:
            continue
    has_final = os.path.exists(os.path.join(checkpoint_dir, f"{name_prefix}_final.zip"))
    n_ckpts = len(steps_files) + (1 if has_final else 0)
    return {"total_steps": best_steps, "checkpoints": n_ckpts}


def best_run(memory_dir: str) -> Optional[Dict]:
    """Highest-scoring iteration from the autonomy / eureka logs."""
    best = None
    for fname in ("autonomy.jsonl", "eureka.jsonl"):
        path = os.path.join(memory_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sc = rec.get("score")
                if sc is None:
                    continue
                if best is None or sc > best["score"]:
                    best = {"score": sc, "source": fname,
                            "iter": rec.get("iter", rec.get("gen")),
                            "metrics": rec.get("metrics", {})}
    return best


def cognition_stats(memory_dir: str) -> Dict:
    """Counts that show how much the agent has accumulated."""
    out = {"events": 0, "lessons": 0, "hypotheses": 0, "experiments": 0,
           "confirmed_hypotheses": 0, "learned_knobs": 0, "frontier_cells": 0,
           "exits_known": 0}

    ev = os.path.join(memory_dir, "episodic", "events.jsonl")
    if os.path.exists(ev):
        with open(ev, encoding="utf-8") as f:
            out["events"] = sum(1 for line in f if line.strip())

    lessons = os.path.join(memory_dir, "lessons", "lessons.jsonl")
    if os.path.exists(lessons):
        with open(lessons, encoding="utf-8") as f:
            out["lessons"] = sum(1 for line in f if line.strip())

    # SQLite-backed counts (built on demand; tolerate absence).
    try:
        from writer import db as _db
        _db.build(memory_dir)
        hyps = _db.query_hypotheses(memory_dir)
        out["hypotheses"] = len(hyps)
        out["confirmed_hypotheses"] = sum(1 for h in hyps if h.get("status") == "confirmed")
        out["experiments"] = len(_db.query_experiments(memory_dir))
    except Exception:
        pass

    try:
        from writer.learned_config import LearnedConfig
        out["learned_knobs"] = len(LearnedConfig(memory_dir).values())
    except Exception:
        pass

    fr = glob.glob(os.path.join(memory_dir, "frontier", "*.json"))
    for f in fr:
        try:
            with open(f, encoding="utf-8") as fh:
                out["frontier_cells"] += len(json.load(fh).get("cells", {}))
        except (json.JSONDecodeError, OSError):
            pass

    out["exits_known"] = len(glob.glob(os.path.join(memory_dir, "exits", "*.json")))
    return out


def disk_usage(cfg) -> Dict:
    """Per-component disk usage of the vault."""
    ckpt = _dir_size(cfg.checkpoint_dir) if os.path.isdir(cfg.checkpoint_dir) else 0
    mem = _dir_size(cfg.memory_dir) if os.path.isdir(cfg.memory_dir) else 0
    vault = _dir_size(cfg.vault_path) if os.path.isdir(cfg.vault_path) else 0
    return {
        "checkpoints_mb": round(ckpt / 1e6, 1),
        "memory_mb": round(mem / 1e6, 1),
        "vault_total_mb": round(vault / 1e6, 1),
    }
