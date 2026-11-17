"""2D LiDAR by ray casting on the ground-truth grid, with noise, dropouts and spurious returns."""
from __future__ import annotations

import numpy as np

from sensors.base_sensor import BaseSensor, SensorObservation


class LidarSensor(BaseSensor):
    name = "lidar"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        c = cfg["sensors"]["lidar"]
        self.beams = int(c["beams"])
        self.max_range = float(c["max_range"])
        self.noise_std = float(c["noise_std"])
        self.dropout = float(c["dropout"])
        self.rel_angles = np.linspace(-np.pi, np.pi, self.beams, endpoint=False)

    def observe(self, world, cond, manual, failed, t, ctx):
        n = self.beams
        if failed:  # dead scanner: no returns at all
            return SensorObservation(self.name, t, True, {"angles": self.rel_angles.copy(),
                                                          "ranges": np.full(n, np.nan),
                                                          "max_range": self.max_range})
        f = world.env.get("lidar_noise", 0.3)
        true = world.cast_rays(world.x, world.y, world.theta + self.rel_angles, self.max_range)
        std = self.noise_std * (1 + 6 * f * (cond["dust"] + 0.5 * cond["smoke"]) + 8 * manual)
        r = true + self.rng.normal(0, std, n)
        p_drop = self.dropout + f * (0.6 * cond["water"] + 0.3 * cond["dust"]) + 0.6 * manual
        p_spur = 0.25 * cond["dust"] * f + 0.1 * manual
        u = self.rng.random(n)
        r = np.where(u < p_drop, np.nan, r)
        spur = (u >= p_drop) & (u < p_drop + p_spur)
        r = np.where(spur, self.rng.uniform(0.3, self.max_range * 0.6, n), r)
        r = np.where(np.isfinite(r), np.clip(r, 0.05, self.max_range), r)
        return SensorObservation(self.name, t, True, {"angles": self.rel_angles.copy(), "ranges": r,
                                                      "max_range": self.max_range})
