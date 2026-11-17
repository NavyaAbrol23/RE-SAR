"""Spatial data association: robot-local candidates -> map frame, then gated nearest-neighbour
assignment to existing tracks. Each track receives at most one candidate per modality per step,
so unrelated positions are never averaged together.
"""
from __future__ import annotations

import math

from sensors.camera_common import EgoGeometry


def to_world(cands: list[dict], pose) -> list[dict]:
    x, y, th = pose
    out = []
    for c in cands:
        wx, wy = EgoGeometry.local_to_world(x, y, th, c["f"], c["l"])
        cc = dict(c, x=wx, y=wy)
        if "bearing" in c:
            cc["bearing_world"] = th + c["bearing"]
        out.append(cc)
    return out


def associate(cands: list[dict], tracks: list, gate: float):
    """Returns (pairs [(cand_idx, track)], unassigned cand indices)."""
    pairs, used, unassigned = [], set(), []
    order = sorted(range(len(cands)), key=lambda i: cands[i]["sigma"])
    for i in order:
        c = cands[i]
        g = min(gate + c["sigma"], 4.0)
        best, bd = None, g
        for t in tracks:
            if (id(t), c["modality"]) in used:
                continue
            d = math.hypot(c["x"] - t.x, c["y"] - t.y)
            if d < bd:
                best, bd = t, d
        if best is None:
            unassigned.append(i)
        else:
            used.add((id(best), c["modality"]))
            pairs.append((i, best))
    return pairs, unassigned
