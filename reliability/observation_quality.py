"""(B) Observation-quality-conditioned reliability: per-observation signal statistics -> q in [0,1].

RGB      q = sqrt(brightness) * (0.5 contrast + 0.5 sharpness-vs-noise)
Thermal  q = (1 - saturated_fraction) * range_factor * noise_factor
LiDAR    q = valid_return_ratio * (1 - adjacent_beam_jump_fraction)
IMU      q = exp(-rms(yaw_rate - commanded_rotation) / 0.08)     (EMA of residual)
Odometry q = exp(-ema|dist - commanded_dist| / 0.35) * exp(-ema|dtheta - cmd| / 0.15)
Audio    q = clip(1 - (noise_floor - 0.02) / 0.45, 0, 1)
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _clip01(v):
    v = float(v)
    if not math.isfinite(v):
        return 0.0
    return min(1.0, max(0.0, v))


def rgb_quality(img: np.ndarray, fov: np.ndarray) -> dict:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    px = g[fov]
    if px.size == 0:
        return {"q": 0.0, "brightness": 0.0, "contrast": 0.0, "sharpness": 0.0}
    mean, std = float(px.mean()), float(px.std())
    lap = cv2.Laplacian(g, cv2.CV_32F)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)      # mostly sensor noise
    noise = float(hp[fov].std())
    structure = float(cv2.GaussianBlur(lap, (0, 0), 1.5)[fov].std())  # edges that survive smoothing
    b = _clip01(mean / 70.0)
    c = _clip01(std / 30.0)
    s = _clip01(structure / (2.0 + 1.5 * noise))
    return {"q": _clip01(math.sqrt(b) * (0.5 * c + 0.5 * s)), "brightness": b, "contrast": c, "sharpness": s}


def thermal_quality(T: np.ndarray, t_sat: float, fov: np.ndarray, t_nominal: float = 120.0) -> dict:
    px = T[fov]
    if px.size == 0:
        return {"q": 0.0, "saturation": 1.0}
    sat = float(np.mean(px >= t_sat - 0.5))
    hp = T - cv2.GaussianBlur(T, (0, 0), 1.2)
    noise = float(hp[fov].std())
    range_f = _clip01((t_sat - 25.0) / (t_nominal - 25.0)) ** 0.5
    noise_f = 1.0 / (1.0 + (noise / 1.0) ** 2)
    return {"q": _clip01((1 - sat) * range_f * (0.4 + 0.6 * noise_f)), "saturation": sat, "noise": noise}


def lidar_quality(ranges: np.ndarray, max_range: float) -> dict:
    n = len(ranges)
    valid = ~np.isnan(ranges)
    vr = float(valid.mean()) if n else 0.0
    fin = np.where(np.isfinite(ranges), ranges, np.nan)
    d = np.abs(np.diff(np.concatenate([fin, fin[:1]])))
    pair = np.isfinite(d)
    jump = float(np.mean(d[pair] > 1.5)) if pair.sum() > 3 else 0.0
    return {"q": _clip01(vr * (1 - jump)), "valid_ratio": vr, "jump_frac": jump}


class ResidualTracker:
    """EMA of command residuals for proprioceptive sensors (IMU / odometry)."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.imu_res2 = 0.0
        self.odo_res = 0.0
        self.odo_rot = 0.0

    def update_imu(self, yaw_rate: float, rot_cmd: float) -> dict:
        r = yaw_rate - rot_cmd if math.isfinite(yaw_rate) else 1.0
        self.imu_res2 = (1 - self.alpha) * self.imu_res2 + self.alpha * r * r
        rms = math.sqrt(self.imu_res2)
        return {"q": _clip01(math.exp(-rms / 0.08)), "residual": rms}

    def update_odo(self, dist: float, dtheta: float, dist_cmd: float, rot_cmd: float) -> dict:
        if not (math.isfinite(dist) and math.isfinite(dtheta)):
            dist, dtheta = 0.0, 0.0
        self.odo_res = (1 - self.alpha) * self.odo_res + self.alpha * abs(dist - dist_cmd)
        self.odo_rot = (1 - self.alpha) * self.odo_rot + self.alpha * abs(dtheta - rot_cmd)
        return {"q": _clip01(math.exp(-self.odo_res / 0.35) * math.exp(-self.odo_rot / 0.15)),
                "residual": self.odo_res}


def audio_quality(noise_floor: float) -> dict:
    return {"q": _clip01(1 - (noise_floor - 0.02) / 0.45), "noise_floor": noise_floor}
