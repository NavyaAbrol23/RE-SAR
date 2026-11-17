"""(E) Shared uncertainty-aware map used by every navigation and RL module.

Layers (all H x W float32, robot's ESTIMATE, never ground truth):
  occupancy      P(occupied) from the log-odds grid
  traversability cost multiplier from sensed terrain
  hazard         sensed hazard intensity in [0,1]
  victim         victim-track confidence splatted as Gaussians
  unexplored     1 where the cell is still unknown
  uncertainty    occupancy entropy + localisation uncertainty at observation time
"""
from __future__ import annotations

import numpy as np

from mapping.occupancy_grid import OccupancyGrid
from mapping.traversability import TerrainEstimator
from mapping.uncertainty_map import UncertaintyMap


class SharedMap:
    def __init__(self, W, H, cfg):
        self.W, self.H = W, H
        self.occ = OccupancyGrid(W, H, cfg)
        self.terrain = TerrainEstimator(W, H)
        self.unc = UncertaintyMap(W, H)
        self.victim = np.zeros((H, W), np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        self._xx, self._yy = xx + 0.5, yy + 0.5

    def reset(self):
        self.occ.reset()
        self.terrain.reset()
        self.unc.reset()
        self.victim[:] = 0

    def update_victims(self, tracks):
        v = np.zeros((self.H, self.W), np.float32)
        for t in tracks:
            s2 = max(min(t.var, 4.0), 0.25)
            v = np.maximum(v, t.prob * np.exp(-((self._xx - t.x) ** 2 + (self._yy - t.y) ** 2) / (2 * s2)))
        self.victim = v

    @property
    def hazard(self):
        return self.terrain.hazard()

    @property
    def unexplored(self):
        return (~self.occ.known).astype(np.float32)

    def layers(self) -> dict:
        p = self.occ.prob
        return {
            "occupancy": p.astype(np.float32),
            "traversability": self.terrain.traversal_cost().astype(np.float32),
            "hazard": self.hazard.astype(np.float32),
            "victim": self.victim,
            "unexplored": self.unexplored,
            "uncertainty": self.unc.layer(p),
        }

    def coverage_estimate(self) -> float:
        return float(self.occ.known.mean())
