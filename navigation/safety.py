"""Safety controller: vetoes a motion step into a known-occupied cell, a sensed impassable
hazard, or toward a close LiDAR return in the direction of travel (only when LiDAR is trusted).
"""
from __future__ import annotations

import math

import numpy as np


class SafetyController:
    def __init__(self, cfg):
        self.impassable = cfg["world"]["impassable_hazard"]
        self.vetoes = 0

    def check(self, shared, est_pose, next_cell, lidar_data, r_lidar: float):
        cx, cy = next_cell
        if not (0 <= cx < shared.W and 0 <= cy < shared.H):
            return False, "out_of_bounds"
        if shared.occ.occupied[cy, cx]:
            self.vetoes += 1
            return False, "occupied"
        if shared.hazard[cy, cx] >= self.impassable:
            self.vetoes += 1
            return False, "hazard"
        if r_lidar > 0.2 and lidar_data is not None:
            x, y, th = est_pose
            tx, ty = cx + 0.5, cy + 0.5
            dist = math.hypot(tx - x, ty - y)
            heading = math.atan2(ty - y, tx - x) - th
            ang = np.angle(np.exp(1j * (lidar_data["angles"] - heading)))
            sel = (np.abs(ang) < math.radians(15)) & np.isfinite(lidar_data["ranges"])
            if sel.any() and float(np.min(lidar_data["ranges"][sel])) < min(dist, 1.0) * 0.6:
                self.vetoes += 1
                return False, "lidar_proximity"
        return True, "ok"
