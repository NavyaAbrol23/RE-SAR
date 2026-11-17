"""IMU: absolute yaw (with bias random walk = drift) and yaw rate, plus noise."""
from __future__ import annotations

import math

import numpy as np

from sensors.base_sensor import BaseSensor, SensorObservation


class IMUSensor(BaseSensor):
    name = "imu"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        c = cfg["sensors"]["imu"]
        self.noise_std, self.bias0, self.drift = c["noise_std"], c["bias"], c["drift"]
        self.reset()

    def reset(self):
        self.bias = self.bias0
        self.yaw_offset = 0.0
        self.last = None

    def observe(self, world, cond, manual, failed, t, ctx):
        if failed:  # frozen output: last value repeated, zero rate
            yaw = self.last["yaw"] if self.last else 0.0
            data = {"yaw": yaw, "yaw_rate": 0.0, "frozen": True}
            return SensorObservation(self.name, t, True, data)
        env = 1 + 3 * (0.15 if world.disaster == "earthquake" else 0.05) * world.severity
        self.bias += self.rng.normal(0, self.drift * (1 + 20 * manual) * env)
        noise = self.noise_std * (1 + 10 * manual) * env
        d_true = ctx.get("actual_rot", 0.0)
        rate = d_true + self.bias + self.rng.normal(0, noise)
        self.yaw_offset += self.bias
        yaw = math.atan2(math.sin(world.theta + self.yaw_offset), math.cos(world.theta + self.yaw_offset))
        yaw += self.rng.normal(0, noise)
        data = {"yaw": float(yaw), "yaw_rate": float(rate), "frozen": False}
        self.last = data
        return SensorObservation(self.name, t, True, data)
