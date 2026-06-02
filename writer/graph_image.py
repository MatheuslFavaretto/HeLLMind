"""Render the vault's knowledge graph as an image — the actual `[[wikilink]]` structure
that Obsidian's Graph View shows, drawn with a small force-directed layout (cv2+numpy, no
new deps). Useful for the README and as a static companion to the live Obsidian graph.

    python -m writer.graph_image            # -> <vault>/attachments/knowledge-graph.png
"""
import glob
import os
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np

_LINK = re.compile(r"\[\[([^\]|#]+)")

# folder prefix -> (node color BGR, ring radius factor)
_TYPE = {
    "00-index": ((230, 210, 120), 0.0),
    "30-runs": ((90, 200, 90), 0.55),
    "10-checkpoints": ((120, 160, 250), 0.95),
    "20-concepts": ((230, 140, 230), 0.75),
    "40-maps": ((90, 220, 220), 0.7),
    "60-lessons": ((120, 120, 240), 0.6),
    "70-bestiary": ((70, 90, 235), 0.65),
}


def _node_type(stem_to_path: Dict[str, str], name: str) -> str:
    p = stem_to_path.get(name, "")
    for t in _TYPE:
        if f"/{t}/" in p or p.startswith(t + "/"):
            return t
    return "20-concepts"


def render_graph(vault_path: str, out_path: str, w: int = 820, h: int = 620) -> bool:
    files = glob.glob(os.path.join(vault_path, "**", "*.md"), recursive=True)
    stem_to_path = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    edges: List[Tuple[str, str]] = []
    nodes = set()
    for f in files:
        src = os.path.splitext(os.path.basename(f))[0]
        try:
            text = open(f, encoding="utf-8").read()
        except OSError:
            continue
        for m in _LINK.findall(text):
            dst = m.strip()
            if dst and dst != src:
                edges.append((src, dst))
                nodes.add(src)
                nodes.add(dst)
    if len(nodes) < 2:
        return False
    nodes = list(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    rng = np.random.default_rng(7)
    pos = rng.uniform(0.2, 0.8, size=(len(nodes), 2))
    E = [(idx[a], idx[b]) for a, b in edges if a in idx and b in idx]
    # tiny force-directed layout
    for _ in range(220):
        disp = np.zeros_like(pos)
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(d, axis=2) + 1e-3
        rep = (d / dist[..., None] ** 2) * 0.0012
        disp += rep.sum(axis=1)
        for a, b in E:
            delta = pos[a] - pos[b]
            disp[a] -= delta * 0.01
            disp[b] += delta * 0.01
        pos += np.clip(disp, -0.03, 0.03)
        pos = np.clip(pos, 0.05, 0.95)

    img = np.full((h, w, 3), 22, np.uint8)
    deg: Dict[int, int] = {}
    for a, b in E:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1

    def P(i):
        return int(pos[i][0] * (w - 80) + 40), int(pos[i][1] * (h - 80) + 40)

    for a, b in E:
        cv2.line(img, P(a), P(b), (70, 70, 75), 1, cv2.LINE_AA)
    for n, i in idx.items():
        color, _ = _TYPE.get(_node_type(stem_to_path, n), ((180, 180, 180), 0.5))
        r = 4 + min(10, deg.get(i, 0))
        cv2.circle(img, P(i), r, color, -1, cv2.LINE_AA)
        if deg.get(i, 0) >= 2:  # label the hubs only (keep it readable)
            label = (n[:22])
            x, y = P(i)
            cv2.putText(img, label, (x + r + 2, y + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(img, "Knowledge graph (vault wikilinks)", (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return bool(cv2.imwrite(out_path, img))


def main() -> None:
    from config import Config
    cfg = Config()
    out = os.path.join(cfg.vault_path, "attachments", "knowledge-graph.png")
    print(f"[graph] wrote {out}" if render_graph(cfg.vault_path, out)
          else "[graph] not enough links yet.")


if __name__ == "__main__":
    main()
