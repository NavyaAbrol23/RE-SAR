"""Costmap from the shared map: occupied / impassable cells -> inf, otherwise
cost = traversability * (1 + hazard_cost * hazard) + unknown_cost * unknown + near-obstacle penalty.
Unknown cells are traversable (optimistic planning) but penalised; the safety controller guards them.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_costmap(shared, cfg) -> np.ndarray:
    n = cfg["navigation"]
    occ = shared.occ.occupied
    haz = shared.hazard
    unknown = ~shared.occ.known
    cost = shared.terrain.traversal_cost() * (1.0 + n["hazard_cost"] * haz) + n["unknown_cost"] * unknown
    near = cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~occ
    cost = cost + 0.6 * near
    cost = cost.astype(np.float64)
    cost[occ] = np.inf
    cost[haz >= cfg["world"]["impassable_hazard"]] = np.inf
    return cost
