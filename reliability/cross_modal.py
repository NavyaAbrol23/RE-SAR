"""(D) Cross-modal agreement, computed ONLY between physically related measurements.

Perception group (victim evidence):
  * RGB <-> thermal: candidates within the association gate (shared field of view).
  * audio <-> {RGB, thermal, tracks}: bearing of an acoustic detection within 0.35 rad of a
    visual/thermal candidate or existing track.
  A modality is only compared with partners that are currently healthy (h > 0.5) and of usable
  quality (q > 0.3), so one failed sensor does not drag down its partners.

Localisation group (motion evidence):
  * odometry displacement  vs  LiDAR scan-matching displacement
  * IMU rotation           vs  LiDAR scan-matching rotation (or vs odometry if no scan match)
  * LiDAR: scan-to-map match score (fraction of endpoints that land on mapped obstacles)

Each agreement is an EMA in [0,1] starting from a neutral 0.5; with no evidence it relaxes
slowly toward 0.5.
"""
from __future__ import annotations

import math

import numpy as np

from sensors.base_sensor import SENSOR_NAMES


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CrossModalAgreement:
    def __init__(self, gate: float = 2.0, alpha: float = 0.25):
        self.gate, self.alpha = gate, alpha
        self.a = {s: 0.5 for s in SENSOR_NAMES}

    def reset(self):
        self.a = {s: 0.5 for s in SENSOR_NAMES}

    def _ema(self, s, v, alpha=None):
        al = self.alpha if alpha is None else alpha
        self.a[s] = float(np.clip((1 - al) * self.a[s] + al * v, 0, 1))

    def _relax(self, s):
        self._ema(s, 0.5, 0.02)

    def update_perception(self, cands: dict, tracks: list, pose, usable: dict):
        """cands: modality -> list of dicts with world 'x','y' (and audio 'bearing_world')."""
        x, y = pose[0], pose[1]
        for m in ("rgb", "thermal"):
            other = "thermal" if m == "rgb" else "rgb"
            if not cands.get(m) or not usable.get(other, False):
                self._relax(m)
                continue
            support = 0
            for c in cands[m]:
                near = any(math.hypot(c["x"] - o["x"], c["y"] - o["y"]) < self.gate for o in cands[other])
                support += near
            self._ema(m, support / len(cands[m]))
        if cands.get("audio") and (usable.get("rgb") or usable.get("thermal") or tracks):
            pts = [(o["x"], o["y"]) for mm in ("rgb", "thermal") if usable.get(mm) for o in cands.get(mm, [])]
            pts += [(t.x, t.y) for t in tracks]
            if pts:
                support = 0
                for c in cands["audio"]:
                    b = c["bearing_world"]
                    support += any(abs(_wrap(math.atan2(py - y, px - x) - b)) < 0.35 for px, py in pts)
                self._ema("audio", support / len(cands["audio"]))
            else:
                self._relax("audio")
        else:
            self._relax("audio")

    def update_localization(self, odo: dict, imu: dict, sm: dict | None, health: dict):
        if sm is not None and sm.get("ok"):
            if health.get("odometry", 0) > 0.5:
                self._ema("odometry", math.exp(-abs(odo["dist"] - sm["dist"]) / 0.25))
            if health.get("imu", 0) > 0.5:
                self._ema("imu", math.exp(-abs(_wrap(imu["yaw_rate"] - sm["dtheta"])) / 0.06))
            self._ema("lidar", float(np.clip((sm["score"] - 0.3) / 0.5, 0, 1)))
        else:
            if health.get("imu", 0) > 0.5 and health.get("odometry", 0) > 0.5:
                v = math.exp(-abs(_wrap(imu["yaw_rate"] - odo["d_theta"])) / 0.08)
                self._ema("imu", v)
                self._ema("odometry", v)
            else:
                self._relax("imu")
                self._relax("odometry")
            self._relax("lidar")
