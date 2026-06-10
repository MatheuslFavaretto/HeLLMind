"""Checkpoint garbage collection — keep `_final` + the newest N step-files per family.

The training loop saves a snapshot every ~50k steps and never deletes (12GB observed).
Resume only ever loads the NEWEST file of a family, so older step-snapshots are dead
weight; a few are kept for `doom-cli progress` curves.

Used by `doom-cli prune` (all families, dry-run by default) and by the autonomous loop
(current family only, after each train chunk — prevents the 12GB recurrence).
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Optional

_STEP_RE = re.compile(r"^(.+?)_(\d+)_steps\.zip$")


def scan_families(dirs: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Map family path-prefix -> [(steps, filepath), ...] for every step-checkpoint.
    `_final.zip` files are never included (always kept)."""
    families: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for d in dict.fromkeys(dirs):  # dedupe, keep order
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = _STEP_RE.match(f)
            if m:
                families[os.path.join(d, m.group(1))].append(
                    (int(m.group(2)), os.path.join(d, f)))
    return families


def prune(dirs: list[str], keep: int = 10, apply: bool = False,
          family: Optional[str] = None) -> tuple[list[str], int]:
    """Delete (or list, when apply=False) step-checkpoints beyond the newest `keep`
    per family. `family` restricts to one brain family (basename prefix match) —
    the autonomous loop uses this so it only ever GCs the brain it is training.

    Returns (paths_pruned_or_prunable, bytes_freed_or_freeable)."""
    if keep <= 0:  # 0/negative = GC disabled
        return [], 0
    victims, freed = [], 0
    for fam, files in sorted(scan_families(dirs).items()):
        if family is not None and os.path.basename(fam) != family:
            continue
        files.sort()  # by step count: oldest first
        for _, path in files[:-keep] if len(files) > keep else []:
            victims.append(path)
            try:
                freed += os.path.getsize(path)
            except OSError:
                continue
    if apply:
        for path in victims:
            try:
                os.remove(path)
            except OSError:
                pass
    return victims, freed


def newest_family(checkpoint_dir: str) -> Optional[str]:
    """Family name (basename prefix) of the most recently written step-checkpoint —
    i.e. the brain currently being trained. None if there are no step files."""
    newest, newest_mtime = None, -1.0
    for fam, files in scan_families([checkpoint_dir]).items():
        for _, path in files:
            try:
                mt = os.path.getmtime(path)
            except OSError:
                continue
            if mt > newest_mtime:
                newest, newest_mtime = os.path.basename(fam), mt
    return newest
