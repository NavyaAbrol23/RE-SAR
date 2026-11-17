"""Hazard / terrain estimation from the RGB and thermal cameras (robot side, no ground truth).

RGB pixels are auto-gained and classified by colour: water (blue dominant), rubble (dark brown),
fire (bright orange-red). Thermal pixels above 45 C, or saturated pixels, indicate heat hazard
(saturated pixels are treated conservatively as hot, 0.75, but not impassable). Per-cell layers are EMA-updated, weighted by the
camera's reliability, and combined into a hazard layer and a traversability cost layer.
"""
from __future__ import annotations

import cv2
import numpy as np


class TerrainEstimator:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.heat = np.zeros((H, W), np.float32)
        self.water = np.zeros((H, W), np.float32)
        self.rubble = np.zeros((H, W), np.float32)
        self.seen = np.zeros((H, W), bool)

    def reset(self):
        for a in (self.heat, self.water, self.rubble):
            a[:] = 0
        self.seen[:] = False

    def _cells(self, geo, pose, mask):
        wx, wy = geo.world_coords(pose[0], pose[1], pose[2])
        cx, cy = np.floor(wx[mask]).astype(int), np.floor(wy[mask]).astype(int)
        ok = (cx >= 0) & (cy >= 0) & (cx < self.W) & (cy < self.H)
        return cx[ok], cy[ok], ok

    def _ema(self, layer, cx, cy, vals, w):
        idx = cy * self.W + cx
        n = np.bincount(idx, minlength=self.W * self.H)
        s = np.bincount(idx, weights=vals, minlength=self.W * self.H)
        has = n > 0
        flat = layer.reshape(-1)
        flat[has] = (1 - w) * flat[has] + w * (s[has] / n[has])

    def update_rgb(self, img, geo, pose, weight):
        if weight <= 0.02:
            return
        fov = geo.fov_mask
        px = img[fov].astype(np.float32)
        if px.size == 0 or px.mean() < 3:
            return
        gain = 200.0 / max(np.percentile(px, 95), 10.0)
        im = np.clip(img.astype(np.float32) * gain, 0, 255)
        r, g, b = im[..., 0], im[..., 1], im[..., 2]
        water = ((b > r + 40) & (b > g + 15)).astype(np.float32)
        fire = ((r > 170) & (g < 120) & (b < 90)).astype(np.float32)
        rub = ((r > g + 12) & (g > b + 12) & (r < 160) & (fire == 0)).astype(np.float32)
        cx, cy, ok = self._cells(geo, pose, fov)
        w = 0.5 * weight
        self._ema(self.water, cx, cy, water[fov][ok], w)
        self._ema(self.rubble, cx, cy, rub[fov][ok], w)
        self._ema(self.heat, cx, cy, fire[fov][ok] * 0.8, w * 0.5)
        self.seen[cy, cx] = True

    def update_thermal(self, T, t_sat, geo, pose, weight):
        if weight <= 0.02:
            return
        fov = geo.fov_mask
        heat = np.clip((T - 45.0) / 75.0, 0, 1)
        heat = np.where(T >= t_sat - 0.5, np.maximum(heat, 0.75), heat).astype(np.float32)  # saturated: hot, but not provably impassable
        cx, cy, ok = self._cells(geo, pose, fov)
        self._ema(self.heat, cx, cy, heat[fov][ok], 0.6 * weight)
        self.seen[cy, cx] = True

    def hazard(self):
        return np.clip(np.maximum.reduce([self.heat, 0.6 * self.water, 0.45 * self.rubble]), 0, 1)

    def traversal_cost(self):
        return 1.0 + 1.5 * self.rubble + 1.0 * self.water + 2.0 * self.heat

    def local(self, x, y, r=2):
        cx, cy = int(x), int(y)
        sl = (slice(max(0, cy - r), cy + r + 1), slice(max(0, cx - r), cx + r + 1))
        return float(self.heat[sl].max(initial=0)), float(self.water[sl].mean()), float(self.rubble[sl].mean())
