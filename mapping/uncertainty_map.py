"""Per-cell uncertainty: occupancy entropy blended with the localisation uncertainty present
when the cell was last observed.   u = 0.5 * H(p) + 0.5 * min(1, sigma_xy_at_observation / 2)
Unknown cells have H = 1 and pose term 1, so u = 1.
"""
from __future__ import annotations

import numpy as np


class UncertaintyMap:
    def __init__(self, W, H):
        self.pose_sigma = np.ones((H, W), np.float32) * 2.0

    def reset(self):
        self.pose_sigma[:] = 2.0

    def observe(self, observed_mask: np.ndarray, sigma_xy: float):
        self.pose_sigma[observed_mask] = 0.7 * self.pose_sigma[observed_mask] + 0.3 * sigma_xy

    def layer(self, prob: np.ndarray) -> np.ndarray:
        p = np.clip(prob, 1e-6, 1 - 1e-6)
        ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return (0.5 * ent + 0.5 * np.clip(self.pose_sigma / 2.0, 0, 1)).astype(np.float32)
