"""Centralised sensor degradation controller.

Combines (a) the environment-induced degradation from the disaster conditions at the robot and
(b) manually injected faults (normal / mild / severe / failed), and exposes the resulting
GROUND-TRUTH effective reliability r* = 1 - e for evaluation only.
"""
from __future__ import annotations

import numpy as np

from sensors.base_sensor import SENSOR_NAMES

LEVELS = ("normal", "mild", "severe", "failed")


class DegradationController:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.level_values = cfg["degradation_levels"]
        self.env = cfg.get("environment", {})
        self.levels = {s: "normal" for s in SENSOR_NAMES}
        self.true_rel = {s: 1.0 for s in SENSOR_NAMES}

    # ------------------------------------------------------------ control
    def set_level(self, sensor: str, level: str):
        if sensor not in SENSOR_NAMES or level not in LEVELS:
            raise ValueError(f"bad sensor/level {sensor}/{level}")
        self.levels[sensor] = level

    def set_levels(self, mapping: dict):
        for s, l in (mapping or {}).items():
            self.set_level(s, l)

    def cycle(self, sensor: str) -> str:
        i = LEVELS.index(self.levels[sensor])
        self.levels[sensor] = LEVELS[(i + 1) % len(LEVELS)]
        return self.levels[sensor]

    def manual(self, sensor: str) -> float:
        return float(self.level_values[self.levels[sensor]])

    def failed(self, sensor: str) -> bool:
        return self.levels[sensor] == "failed"

    # ------------------------------------------------------- ground truth
    def env_degradation(self, sensor: str, c: dict, severity: float) -> float:
        e = self.env
        if sensor == "rgb":
            v = 1 - (1 - 0.9 * c["smoke"]) * (1 - c["darkness"]) * (1 - 0.7 * c["dust"])
        elif sensor == "thermal":
            v = 0.85 * c["heat"] * e.get("heat", 0) + 0.15 * c["smoke"]
        elif sensor == "lidar":
            v = e.get("lidar_noise", 0.3) * (c["dust"] + 0.5 * c["smoke"] + 0.9 * c["water"]) + 0.05 * severity
        elif sensor == "imu":
            v = (0.15 if self.cfg.get("disaster") == "earthquake" else 0.05) * severity
        elif sensor == "odometry":
            v = 1.6 * c["slip"]
        elif sensor == "audio":
            v = 0.7 * c["acoustic_noise"] + 0.4 * e.get("audio_attenuation", 0) * severity
        else:
            v = 0.0
        return float(np.clip(v, 0.0, 0.95))

    def update_truth(self, cond: dict, severity: float) -> dict:
        for s in SENSOR_NAMES:
            if self.failed(s):
                self.true_rel[s] = 0.0
            else:
                e = 1 - (1 - self.env_degradation(s, cond, severity)) * (1 - self.manual(s))
                self.true_rel[s] = float(np.clip(1 - e, 0.0, 1.0))
        return dict(self.true_rel)
