"""Wheel odometry: forward displacement + heading change, corrupted by wheel slip."""
from __future__ import annotations

import numpy as np

from sensors.base_sensor import BaseSensor, SensorObservation


class OdometrySensor(BaseSensor):
    name = "odometry"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        c = cfg["sensors"]["odometry"]
        self.noise_std, self.slip_scale = c["noise_std"], c["slip_scale"]
        self.reset()

    def reset(self):
        self.acc_err = 0.0

    def observe(self, world, cond, manual, failed, t, ctx):
        if failed:  # dead encoders
            return SensorObservation(self.name, t, True, {"dist": 0.0, "d_theta": 0.0})
        d_cmd = ctx.get("dist_cmd", 0.0)
        d_act = ctx.get("actual_dist", 0.0)
        noise = self.noise_std * (1 + 6 * manual)
        if ctx.get("slipped", False) or ctx.get("collided", False):
            dist = d_cmd * (1 + self.rng.normal(0, 0.05))  # wheels turned, robot didn't move
        else:
            dist = d_act * (1 + self.slip_scale * manual + self.rng.normal(0, noise)) if d_act > 0 else 0.0
        dth = ctx.get("actual_rot", 0.0) + self.rng.normal(0, 0.02 * (1 + 5 * cond["slip"] + 6 * manual))
        self.acc_err += abs(dist - d_act)
        return SensorObservation(self.name, t, True, {"dist": float(max(dist, 0.0)), "d_theta": float(dth)})
