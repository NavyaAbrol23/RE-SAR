"""Sensor suite: all six sensors + the centralised degradation controller."""
from __future__ import annotations

import numpy as np

from sensors.audio import AudioSensor
from sensors.degradation import DegradationController
from sensors.imu import IMUSensor
from sensors.lidar import LidarSensor
from sensors.odometry import OdometrySensor
from sensors.rgb import RGBCamera
from sensors.thermal import ThermalCamera


class SensorSuite:
    def __init__(self, cfg: dict, seed: int):
        self.rng = np.random.default_rng(seed + 7919)
        self.degradation = DegradationController(cfg)
        self.sensors = {
            "rgb": RGBCamera(cfg, self.rng),
            "thermal": ThermalCamera(cfg, self.rng),
            "lidar": LidarSensor(cfg, self.rng),
            "imu": IMUSensor(cfg, self.rng),
            "odometry": OdometrySensor(cfg, self.rng),
            "audio": AudioSensor(cfg, self.rng),
        }

    def reset(self):
        for s in self.sensors.values():
            s.reset()

    def observe(self, world, t: int, ctx: dict) -> dict:
        cond = world.ground_truth_conditions()
        self.degradation.update_truth(cond, world.severity)
        out = {}
        for name, s in self.sensors.items():
            out[name] = s.observe(world, cond, self.degradation.manual(name),
                                  self.degradation.failed(name), t, ctx)
        return out

    @property
    def true_reliability(self) -> dict:
        """GROUND TRUTH — evaluation only."""
        return dict(self.degradation.true_rel)
