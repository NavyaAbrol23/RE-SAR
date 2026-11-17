"""(A) Environment-conditioned reliability prior.

The robot does NOT read ground-truth smoke/heat/water fields. It predicts local conditions from
  * the mission profile (disaster type + severity, assumed known to the operator), and
  * its own sensed hazard layers in the shared map at its ESTIMATED position,
then maps predicted conditions to a per-sensor reliability prior:

    r_env[s] = clip(1 - D_s(c_hat), 0, 1)

with the degradation functions D_s defined below (same functional family as the simulator, but
fed with estimated, not true, conditions). It cannot see injected sensor faults by design.
"""
from __future__ import annotations

import numpy as np

from sensors.base_sensor import SENSOR_NAMES


class EnvironmentStateEstimator:
    def __init__(self, cfg: dict):
        self.env = cfg.get("environment", {})
        self.disaster = cfg["disaster"]
        self.severity = float(cfg["world"]["severity"])
        self.last_conditions: dict = {}

    def estimate_conditions(self, local_heat: float, local_water: float, local_rubble: float) -> dict:
        e, s = self.env, self.severity
        c = {
            "smoke": float(np.clip(e.get("smoke", 0) * ((0.4 + 0.6 * s) * min(1.0, 2 * local_heat) + 0.15 * s), 0, 1)),
            "dust": float(np.clip(e.get("dust", 0) * (0.3 * s + 0.9 * s * local_rubble), 0, 1)),
            "darkness": float(np.clip(e.get("darkness", 0) * (0.5 + 0.5 * s), 0, 0.95)),
            "heat": float(np.clip(local_heat, 0, 1)),
            "water": float(np.clip(local_water, 0, 1)),
            "acoustic_noise": float(np.clip(e.get("acoustic_noise", 0.3) * (0.5 + 0.7 * s), 0, 1)),
            "slip": float(e.get("wheel_slip", 0.1) * (0.5 + s) * (0.15 + 0.85 * max(local_water > 0.1, local_rubble))),
        }
        self.last_conditions = c
        return c

    def reliability(self, c: dict) -> dict:
        e, s = self.env, self.severity
        d = {
            "rgb": 1 - (1 - 0.9 * c["smoke"]) * (1 - c["darkness"]) * (1 - 0.7 * c["dust"]),
            "thermal": 0.85 * c["heat"] * e.get("heat", 0) + 0.15 * c["smoke"],
            "lidar": e.get("lidar_noise", 0.3) * (c["dust"] + 0.5 * c["smoke"] + 0.9 * c["water"]) + 0.05 * s,
            "imu": (0.15 if self.disaster == "earthquake" else 0.05) * s,
            "odometry": 1.6 * c["slip"],
            "audio": 0.7 * c["acoustic_noise"] + 0.4 * e.get("audio_attenuation", 0) * s,
        }
        return {k: float(np.clip(1 - np.clip(d[k], 0, 0.95), 0, 1)) for k in SENSOR_NAMES}
