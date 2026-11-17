"""Log-odds occupancy grid updated from LiDAR scans at the ESTIMATED pose.

Each valid beam passes through cells (free evidence) and, if it hit something within max range,
ends in a cell (occupied evidence). Per scan, a cell receives l_occ if at least as many beams end
in it as pass through it, otherwise l_free (robust to grazing hits at door jambs and corners). Increments are multiplied by a weight (the LiDAR reliability in
reliability-aware modes, 1.0 in fixed mode). NaN beams (dropouts) are ignored.
"""
from __future__ import annotations

import numpy as np


class OccupancyGrid:
    def __init__(self, width: int, height: int, cfg: dict):
        m = cfg["mapping"]
        self.W, self.H = width, height
        self.l_occ, self.l_free = m["l_occ"], m["l_free"]
        self.l_min, self.l_max = m["l_min"], m["l_max"]
        self.known_thr = m["known_threshold"]
        self.occ_p, self.free_p = m["occ_prob"], m["free_prob"]
        self.L = np.zeros((height, width), dtype=np.float32)
        self.last_idx = np.zeros(0, dtype=int)

    def reset(self):
        self.L[:] = 0

    @property
    def prob(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.L))

    @property
    def known(self) -> np.ndarray:
        return np.abs(self.L) > self.known_thr

    @property
    def occupied(self) -> np.ndarray:
        return self.prob > self.occ_p

    @property
    def free(self) -> np.ndarray:
        return self.prob < self.free_p

    def entropy(self) -> float:
        p = np.clip(self.prob, 1e-6, 1 - 1e-6)
        return float(np.sum(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))

    def endpoints(self, pose, angles, ranges):
        x, y, th = pose
        ok = np.isfinite(ranges)
        a = th + angles[ok]
        return x + ranges[ok] * np.cos(a), y + ranges[ok] * np.sin(a)

    def update(self, pose, angles, ranges, max_range, weight: float = 1.0, step: float = 0.4):
        self.last_idx = np.zeros(0, dtype=int)
        if weight <= 0:
            return 0
        x, y, th = pose
        valid = ~np.isnan(ranges)
        if not valid.any():
            return 0
        r = np.where(np.isinf(ranges), max_range, ranges)[valid]
        hit = np.isfinite(ranges[valid])
        a = th + angles[valid]
        ts = np.arange(0.0, max_range, step)
        frac = ts[None, :]
        inside = frac < (r[:, None] - 0.3)
        px = x + np.cos(a)[:, None] * frac
        py = y + np.sin(a)[:, None] * frac
        cx, cy = np.floor(px[inside]).astype(int), np.floor(py[inside]).astype(int)
        ib = (cx >= 0) & (cy >= 0) & (cx < self.W) & (cy < self.H)
        n = self.W * self.H
        # pass-through count: number of distinct beams traversing each cell in this scan
        beam_id = np.broadcast_to(np.arange(len(r))[:, None], inside.shape)[inside][ib]
        pair = np.unique(beam_id * n + cy[ib] * self.W + cx[ib])
        passes = np.bincount(pair % n, minlength=n)
        rh = r[hit] + 0.1  # beams hit wall FACES; push slightly inside so noise doesn't thicken walls
        hx = np.floor(x + rh * np.cos(a[hit])).astype(int)
        hy = np.floor(y + rh * np.sin(a[hit])).astype(int)
        ib2 = (hx >= 0) & (hy >= 0) & (hx < self.W) & (hy < self.H)
        hits = np.bincount(hy[ib2] * self.W + hx[ib2], minlength=n)
        # per-scan vote: a cell is occupied evidence only if more beams end in it than pass through
        hit_idx = np.nonzero((hits > 0) & (hits >= passes))[0]
        free_idx = np.nonzero((passes > 0) & (passes > hits))[0]
        self.last_idx = np.concatenate([free_idx, hit_idx])
        flat = self.L.reshape(-1)
        before = np.abs(flat) > self.known_thr
        n_before = int(before.sum())
        flat[free_idx] += weight * self.l_free
        flat[hit_idx] += weight * self.l_occ
        np.clip(flat, self.l_min, self.l_max, out=flat)
        return int((np.abs(flat) > self.known_thr).sum()) - n_before
