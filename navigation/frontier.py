"""Frontier detection, clustering, feature extraction and heuristic selection.

Frontier cell = known-free cell with at least one unknown 4-neighbour.
Frontiers are clustered (8-connectivity); each cluster's target is its member nearest the centroid.
Candidates are ORDERED by path cost (nearest first); this ordering defines PPO action indices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frontier:
    x: int
    y: int
    size: int
    dist: float        # path cost from robot (Dijkstra)
    info_gain: float   # unknown cells within ig_radius
    hazard: float      # sensed hazard at target
    victim: float      # victim-layer confidence near target
    uncertainty: float # mean uncertainty near target


def frontier_mask(shared) -> np.ndarray:
    known = shared.occ.known
    free = shared.occ.free & known
    unk = (~known).astype(np.uint8)
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
    adj = cv2.dilate(unk, k).astype(bool)
    return free & adj


def find_frontiers(shared, dist_map: np.ndarray, cfg, blacklist=()) -> list[Frontier]:
    n = cfg["navigation"]
    fm = frontier_mask(shared)
    if not fm.any():
        return []
    r = int(n["ig_radius"])
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)).astype(np.float32)
    unk = shared.unexplored
    ig = cv2.filter2D(unk, -1, disk, borderType=cv2.BORDER_CONSTANT)
    layers_unc = shared.unc.layer(shared.occ.prob)
    haz = shared.hazard
    vic = shared.victim
    num, lab = cv2.connectedComponents(fm.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, num):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 2:
            continue
        d = dist_map[ys, xs]
        reach = np.isfinite(d)
        if not reach.any():
            continue
        xs, ys, d = xs[reach], ys[reach], d[reach]
        cx, cy = xs.mean(), ys.mean()
        j = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2 + 0.01 * d))
        tx, ty = int(xs[j]), int(ys[j])
        if (tx, ty) in blacklist:
            continue
        sl = (slice(max(0, ty - 2), ty + 3), slice(max(0, tx - 2), tx + 3))
        out.append(Frontier(tx, ty, len(xs), float(d[j]), float(ig[ty, tx]), float(haz[ty, tx]),
                            float(vic[sl].max()), float(layers_unc[sl].mean())))
    out.sort(key=lambda f: f.dist)
    return out


def select_nearest(frontiers, rng=None):
    return 0 if frontiers else None


def select_random(frontiers, rng):
    return int(rng.integers(len(frontiers))) if frontiers else None


def select_infogain(frontiers, cfg, rng=None):
    if not frontiers:
        return None
    lam = cfg["navigation"]["ig_lambda"]
    util = [(f.info_gain + 15.0 * f.victim) * math.exp(-lam * f.dist) * (1 - 0.7 * f.hazard) for f in frontiers]
    return int(np.argmax(util))
