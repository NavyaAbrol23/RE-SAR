"""(C) Sensor health: availability, invalid readings, frozen/dead outputs, abnormal values.

Returns h in [0,1] per sensor plus a label OK / DEGRADED / FAILED. h = 0 is a hard gate:
a failed sensor contributes nothing to fusion.
"""
from __future__ import annotations

import math

import numpy as np


def label(h: float) -> str:
    return "FAILED" if h <= 0.05 else ("DEGRADED" if h < 0.7 else "OK")


class SensorHealthMonitor:
    def __init__(self, persistence: int = 1):
        # 1 step suffices: a healthy encoder never reads exactly 0 while commanded to drive, and a
        # healthy IMU never repeats a bit-identical yaw while commanded to turn.
        self.k = persistence
        self.prev_rgb = None
        self.imu_frozen = 0
        self.odo_dead = 0
        self.last_yaw = None

    def reset(self):
        self.__init__(self.k)

    def assess(self, obs: dict, cmd: dict) -> dict:
        h = {}
        # RGB: black frame or frozen frame
        o = obs["rgb"]
        if not o.delivered:
            h["rgb"] = 0.0
        else:
            img = o.data["image"][o.data["fov_mask"]]
            mean = float(img.mean()) if img.size else 0.0
            frozen = self.prev_rgb is not None and np.array_equal(self.prev_rgb, o.data["image"])
            self.prev_rgb = o.data["image"]
            h["rgb"] = 0.0 if (mean < 3.0 or frozen) else min(1.0, mean / 15.0)
        # Thermal: saturated fraction
        o = obs["thermal"]
        if not o.delivered:
            h["thermal"] = 0.0
        else:
            px = o.data["temp"][o.data["fov_mask"]]
            sat = float(np.mean(px >= o.data["t_sat"] - 0.5)) if px.size else 1.0
            bad = not np.all(np.isfinite(px))
            h["thermal"] = 0.0 if (sat > 0.95 or bad) else float(np.clip(1 - max(0.0, sat - 0.4) * 1.8, 0, 1))
        # LiDAR: missing returns
        o = obs["lidar"]
        if not o.delivered:
            h["lidar"] = 0.0
        else:
            nan_frac = float(np.mean(np.isnan(o.data["ranges"])))
            h["lidar"] = 0.0 if nan_frac > 0.95 else float(np.clip(1 - nan_frac * 1.2, 0, 1))
        # IMU: frozen yaw while the robot is commanded to rotate, NaN, abnormal rate
        o = obs["imu"]
        yaw, rate = o.data.get("yaw", float("nan")), o.data.get("yaw_rate", float("nan"))
        if not o.delivered or not (math.isfinite(yaw) and math.isfinite(rate)):
            h["imu"] = 0.0
        else:
            if self.last_yaw is not None and yaw == self.last_yaw and abs(cmd.get("rot_cmd", 0)) > 1e-3:
                self.imu_frozen += 1
            elif self.last_yaw is not None and yaw != self.last_yaw:
                self.imu_frozen = 0
            self.last_yaw = yaw
            h["imu"] = 0.0 if self.imu_frozen >= self.k else (0.3 if abs(rate) > 3.5 else 1.0)
        # Odometry: exactly-zero readings while commanded to drive or turn (dead encoders)
        o = obs["odometry"]
        dist, dth = o.data.get("dist", float("nan")), o.data.get("d_theta", float("nan"))
        if not o.delivered or not (math.isfinite(dist) and math.isfinite(dth)):
            h["odometry"] = 0.0
        else:
            moving = cmd.get("dist_cmd", 0) > 0.2
            turning = abs(cmd.get("rot_cmd", 0)) > 1e-3
            if (moving and dist == 0.0) or (turning and dth == 0.0):
                self.odo_dead += 1
            elif dist != 0.0 or dth != 0.0:
                self.odo_dead = 0
            h["odometry"] = 0.0 if self.odo_dead >= self.k else 1.0
        # Audio: saturated microphone / burst of detections
        o = obs["audio"]
        if not o.delivered:
            h["audio"] = 0.0
        else:
            nf = o.data["noise_floor"]
            nd = len(o.data["detections"])
            h["audio"] = 0.0 if nf > 1.0 else (0.4 if nd > 3 else 1.0)
        return {k: float(np.clip(v, 0, 1)) for k, v in h.items()}
