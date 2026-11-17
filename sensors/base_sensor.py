"""Standardised sensor observation format and sensor base class."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SENSOR_NAMES = ("rgb", "thermal", "lidar", "imu", "odometry", "audio")
PERCEPTION_SENSORS = ("rgb", "thermal", "audio")      # victim perception group
LOCALIZATION_SENSORS = ("lidar", "imu", "odometry")   # localisation / mapping group


@dataclass
class SensorObservation:
    """What the robot receives from one sensor at one timestep.

    `data` holds the measurement; `delivered` is False if no message arrived at all.
    Ground-truth reliability is NOT stored here (it lives in the DegradationController and
    is used only for evaluation), so the robot pipeline cannot accidentally read it.
    """
    name: str
    t: int
    delivered: bool
    data: dict[str, Any] = field(default_factory=dict)


class BaseSensor:
    name = "base"

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def reset(self):
        pass

    def observe(self, world, cond: dict, manual: float, failed: bool, t: int, ctx: dict) -> SensorObservation:
        raise NotImplementedError
