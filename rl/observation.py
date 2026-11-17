"""Structured, normalised observation vector for the PPO agent (MlpPolicy).

Built ONLY from the robot's own estimates (EKF pose, shared map, frontier list, reliability
scores). No ground-truth victim/obstacle/pose information is used.

Layout (228 floats, all clipped to [-1, 1]):
  [0:6]     robot: x/W, y/H, cos th, sin th, min(sigma_xy, 2)/2, t / max_steps
  [6:11]    global: coverage estimate, confirmed victims / expected, #frontiers / K,
            sensed hazard at robot, victim-layer max near robot
  [11:17]   reliability r_s for rgb, thermal, lidar, imu, odometry, audio
  [17:164]  local 14x14 window around the robot, 2x2 average-pooled to 7x7, 3 channels:
            occupancy probability, unknown mask, sensed hazard
  [164:228] K=8 frontier slots x 8 features (ordered by A*/Dijkstra path cost, nearest first):
            valid, dx/W, dy/H, path_cost/60, info_gain/50, hazard, victim, uncertainty
"""
from __future__ import annotations

import numpy as np

from sensors.base_sensor import SENSOR_NAMES

K_FRONTIERS = 8
WIN = 14
POOL = 2
OBS_DIM = 6 + 5 + 6 + 3 * (WIN // POOL) ** 2 + K_FRONTIERS * 8


def _window(layer, cx, cy, fill):
    H, W = layer.shape
    half = WIN // 2
    out = np.full((WIN, WIN), fill, np.float32)
    x0, y0 = cx - half, cy - half
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + WIN), min(H, y0 + WIN)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = layer[ys0:ys1, xs0:xs1]
    return out.reshape(WIN // POOL, POOL, WIN // POOL, POOL).mean(axis=(1, 3))


def build_observation(mission) -> np.ndarray:
    W, H = mission.world.W, mission.world.H
    x, y, th = mission.ekf.pose
    cx, cy = mission.est_cell()
    shared = mission.shared
    fr = mission.frontiers
    n_exp = max(1, int(mission.cfg["world"]["num_victims"]))
    haz = shared.hazard
    sl = (slice(max(0, cy - 2), cy + 3), slice(max(0, cx - 2), cx + 3))
    v = [x / W, y / H, np.cos(th), np.sin(th), min(mission.ekf.sigma_xy, 2.0) / 2.0,
         mission.t / mission.cfg["robot"]["max_steps"],
         shared.coverage_estimate(), min(len(mission.tracker.confirmed) / n_exp, 1.0),
         len(fr) / K_FRONTIERS, float(haz[cy, cx]), float(shared.victim[sl].max())]
    v += [mission.rel.r[s] for s in SENSOR_NAMES]
    v = np.array(v, np.float32)
    loc = np.concatenate([_window(shared.occ.prob, cx, cy, 1.0).ravel(),
                          _window(shared.unexplored, cx, cy, 0.0).ravel(),
                          _window(haz, cx, cy, 0.0).ravel()])
    f = np.zeros((K_FRONTIERS, 8), np.float32)
    for i, fe in enumerate(fr[:K_FRONTIERS]):
        f[i] = [1.0, (fe.x + 0.5 - x) / W, (fe.y + 0.5 - y) / H, fe.dist / 60.0, fe.info_gain / 50.0,
                fe.hazard, fe.victim, fe.uncertainty]
    obs = np.concatenate([v, loc, f.ravel()]).astype(np.float32)
    return np.clip(np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
