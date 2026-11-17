"""Lightweight reliability-weighted EKF localisation.

State  mu = [x, y, theta]  (map frame = world frame; the start pose is known).

Prediction (control = fused motion increment):
    dtheta = (r_imu * imu_rate + r_odo * odo_dtheta + c * rot_cmd) / (r_imu + r_odo + c)
    d      = (r_odo * odo_dist + c * dist_cmd) / (r_odo + c)
    theta' = theta + dtheta ;  x' = x + d cos theta' ;  y' = y + d sin theta'
  c = command prior (small constant): commanded motion is the last-resort fallback when all
  proprioceptive sensors have failed. Process noise grows as the motion reliabilities drop.

Update (pose pseudo-measurement from LiDAR scan matching, H = I):
    R = diag(s_xy^2, s_xy^2, s_th^2) / max(r_lidar * match_quality, 0.05)
In `fixed` mode all r = 1, i.e. a conventional EKF with constant noise.
"""
from __future__ import annotations

import math

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class EKFLocalizer:
    def __init__(self, start_pose, cfg: dict):
        self.cmd_prior = cfg["reliability"]["command_prior"]
        self.reset(start_pose)

    def reset(self, start_pose):
        self.mu = np.array(start_pose, dtype=np.float64)
        self.P = np.diag([1e-4, 1e-4, 1e-4])
        self.last_dtheta = 0.0
        self.last_dist = 0.0

    @property
    def pose(self):
        return self.mu.copy()

    @property
    def sigma_xy(self) -> float:
        return float(math.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0)))

    def predict(self, odo: dict, imu: dict, cmd: dict, r: dict):
        c = self.cmd_prior
        r_imu, r_odo = max(r.get("imu", 0.0), 0.0), max(r.get("odometry", 0.0), 0.0)
        imu_rate = imu.get("yaw_rate", 0.0) if math.isfinite(imu.get("yaw_rate", float("nan"))) else 0.0
        odo_dth = odo.get("d_theta", 0.0) if math.isfinite(odo.get("d_theta", float("nan"))) else 0.0
        odo_d = odo.get("dist", 0.0) if math.isfinite(odo.get("dist", float("nan"))) else 0.0
        dth = (r_imu * imu_rate + r_odo * odo_dth + c * cmd["rot_cmd"]) / (r_imu + r_odo + c)
        d = (r_odo * odo_d + c * cmd["dist_cmd"]) / (r_odo + c)
        th = wrap(self.mu[2] + dth)
        self.mu = np.array([self.mu[0] + d * math.cos(th), self.mu[1] + d * math.sin(th), th])
        G = np.array([[1, 0, -d * math.sin(th)], [0, 1, d * math.cos(th)], [0, 0, 1]])
        s_d = 0.06 + 0.25 * (1 - min(r_odo, 1.0)) * max(d, cmd["dist_cmd"])
        s_t = 0.01 + 0.12 * (1 - min(max(r_imu, r_odo), 1.0)) * (abs(dth) + 0.2)
        V = np.array([[math.cos(th), -d * math.sin(th)], [math.sin(th), d * math.cos(th)], [0, 1]])
        Q = V @ np.diag([s_d ** 2, s_t ** 2]) @ V.T + np.diag([1e-4, 1e-4, 1e-5])
        self.P = G @ self.P @ G.T + Q
        self.last_dtheta, self.last_dist = dth, d

    def update(self, z_pose, r_lidar: float, match_quality: float):
        w = max(r_lidar * match_quality, 0.0)
        if w < 0.05:
            return False
        R = np.diag([0.05 ** 2, 0.05 ** 2, 0.02 ** 2]) / w
        y = np.array(z_pose, float) - self.mu
        y[2] = wrap(y[2])
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.mu = self.mu + K @ y
        self.mu[2] = wrap(self.mu[2])
        self.P = (np.eye(3) - K) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return True
