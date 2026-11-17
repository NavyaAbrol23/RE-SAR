"""Disaster-specific hazard intensity, terrain classes and environmental degradation fields."""
from __future__ import annotations

import cv2
import numpy as np

from environment.terrain import BURNING, FLOOR, RUBBLE, WATER


def _smooth_noise(shape, rng, sigma):
    n = rng.random(shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    n -= n.min()
    return n / max(n.max(), 1e-6)


def _blur(field, sigma):
    return cv2.GaussianBlur(field.astype(np.float32), (0, 0), sigma)


def generate_hazards(grid: np.ndarray, rubble: np.ndarray, start, disaster: str, severity: float,
                     env: dict, rng: np.random.Generator) -> dict:
    """Return dict of ground-truth fields (all float32 in [0,1] unless noted)."""
    H, W = grid.shape
    sx, sy = start
    yy, xx = np.mgrid[0:H, 0:W]
    dist_start = np.hypot(xx - sx, yy - sy)
    terrain = np.zeros((H, W), dtype=np.int8)
    hazard = np.zeros((H, W), dtype=np.float32)
    heat = np.zeros((H, W), dtype=np.float32)
    water = np.zeros((H, W), dtype=np.float32)
    free = ~grid

    terrain[rubble & free] = RUBBLE
    hazard[rubble & free] = 0.25 + 0.3 * severity

    if disaster == "fire":
        n_fires = 2 + int(round(3 * severity))
        cand = np.argwhere(free & (dist_start > 8))
        for idx in rng.choice(len(cand), size=min(n_fires, len(cand)), replace=False):
            fy, fx = cand[idx]
            r = 1.5 + 2.5 * severity * rng.random() + 1.0
            heat = np.maximum(heat, np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * r * r)))
        heat *= 0.75 + 0.35 * severity   # fire cores become impassable at medium+ severity
        heat = np.clip(heat, 0, 1)
        burning = (heat > 0.55) & free
        terrain[burning] = BURNING
        hazard = np.maximum(hazard, np.where(free, heat, 0))
    elif disaster == "flood":
        low = _smooth_noise((H, W), rng, sigma=4.0)
        low = 0.6 * low + 0.4 * (xx / W)          # water pools toward the far side
        coverage = env.get("water", 0.8) * (0.25 + 0.45 * severity)
        thr = np.quantile(low, 1 - coverage)
        depth = np.clip((low - thr) / max(1 - thr, 1e-6), 0, 1)
        depth[dist_start < 5] = 0
        water = np.where(free, depth, 0).astype(np.float32)
        wet = water > 0.05
        terrain[wet] = WATER
        hazard = np.maximum(hazard, 0.2 + 0.8 * water * wet)
        hazard[~wet & (terrain != RUBBLE)] = 0
    elif disaster == "earthquake":
        unstable = _smooth_noise((H, W), rng, 3.0) > (0.75 - 0.15 * severity)
        unstable &= free & (dist_start > 5)
        terrain[unstable & (terrain == FLOOR)] = RUBBLE
        hazard[unstable] = np.maximum(hazard[unstable], 0.3 + 0.4 * severity)

    hazard[dist_start < 3] = 0
    terrain[dist_start < 3] = FLOOR
    hazard = np.clip(np.where(free, hazard, 0), 0, 1).astype(np.float32)

    # Environmental degradation fields (affect sensors, not traversal)
    smoke = np.clip(_blur(heat, 4.0) * 2.0, 0, 1) * env.get("smoke", 0.0) * (0.4 + 0.6 * severity)
    smoke = np.clip(smoke + env.get("smoke", 0.0) * 0.15 * severity, 0, 1)
    dust_local = _blur((terrain == RUBBLE).astype(np.float32), 2.5)
    dust = np.clip(env.get("dust", 0.0) * (0.3 * severity + 0.9 * severity * dust_local), 0, 1)
    return {
        "terrain": terrain,
        "hazard": hazard,
        "heat": heat.astype(np.float32),
        "water": water.astype(np.float32),
        "smoke": smoke.astype(np.float32),
        "dust": dust.astype(np.float32),
    }
