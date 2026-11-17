"""Basic correlative scan-to-map matching (endpoint likelihood field + free-space check).

Occupied cells (P > 0.6) are rasterised at x4 resolution. The endpoint field is
    f(p) = exp(-d(p)^2 / (2 * 0.2^2)),  d = distance from p to the nearest occupied region,
so a beam ending exactly on a wall FACE scores 1 (cells are 1 x 1, beams hit faces, not centres).
A point 0.35 cell before each endpoint must be free; landing inside an obstacle is penalised:
    score = mean f(endpoints) - 0.5 * mean occ(pre-endpoints)
Two-stage search around the EKF prediction: coarse (+/-0.75..1.5 cell, +/-0.12..0.35 rad, scaled by
the EKF covariance), then fine.
"""
from __future__ import annotations

import cv2
import numpy as np

UP = 4


def build_fields(prob: np.ndarray):
    occ = (prob > 0.6).astype(np.uint8)
    big = cv2.resize(occ, (occ.shape[1] * UP, occ.shape[0] * UP), interpolation=cv2.INTER_NEAREST)
    d = cv2.distanceTransform((1 - big).astype(np.uint8), cv2.DIST_L2, 3) / UP
    return np.exp(-(d ** 2) / (2 * 0.2 ** 2)).astype(np.float32), big.astype(np.float32)


def _sample(field, x, y):
    H, W = field.shape
    fx = np.clip(x * UP - 0.5, 0, W - 1.001)
    fy = np.clip(y * UP - 0.5, 0, H - 1.001)
    x0, y0 = fx.astype(int), fy.astype(int)
    ax, ay = fx - x0, fy - y0
    return (field[y0, x0] * (1 - ax) * (1 - ay) + field[y0, x0 + 1] * ax * (1 - ay)
            + field[y0 + 1, x0] * (1 - ax) * ay + field[y0 + 1, x0 + 1] * ax * ay)


def _nearest(img, x, y):
    H, W = img.shape
    return img[np.clip((y * UP).astype(int), 0, H - 1), np.clip((x * UP).astype(int), 0, W - 1)]


def _search(fields, x, y, th, a, r, hit, xy, tw, nxy, nt):
    f_end, occ = fields
    dxs = np.linspace(-xy, xy, nxy)
    dts = np.linspace(-tw, tw, nt)
    gx, gy, gt = [g.ravel() for g in np.meshgrid(dxs, dxs, dts, indexing="ij")]
    ang = th + gt[:, None] + a[None, :]
    c, s = np.cos(ang), np.sin(ang)
    bx, by = x + gx[:, None], y + gy[:, None]
    rr = np.minimum(r, 1e9)
    score = np.zeros(len(gx))
    if hit.any():
        ex, ey = bx + rr[None, hit] * c[:, hit], by + rr[None, hit] * s[:, hit]
        score += _sample(f_end, ex, ey).mean(axis=1)
    pre = np.maximum(rr - 0.35, 0.1)
    px, py = bx + pre[None, :] * c, by + pre[None, :] * s
    score -= 0.5 * _nearest(occ, px, py).mean(axis=1)
    score -= 0.01 * np.hypot(gx, gy) / max(xy, 1e-6)
    b = int(np.argmax(score))
    return x + gx[b], y + gy[b], th + gt[b], float(score[b])


def scan_match(prob: np.ndarray, known_occ_count: int, pose, angles, ranges, max_range=8.0,
               min_points=12, fields=None, sigma_xy: float = 0.0, sigma_th: float = 0.0):
    x, y, th = pose
    ok = ~np.isnan(ranges)
    hit = np.isfinite(ranges[ok])
    if hit.sum() < min_points or known_occ_count < 25:
        return {"ok": False, "pose": np.array(pose, float), "score": 0.0}
    a = angles[ok]
    r = np.where(np.isfinite(ranges[ok]), ranges[ok], max_range)
    fields = build_fields(prob) if fields is None else fields
    # search window grows with the EKF's own uncertainty (e.g. after a motion-sensor failure)
    xy = float(np.clip(2.5 * sigma_xy, 0.75, 1.5))
    tw = float(np.clip(2.5 * sigma_th, 0.12, 0.35))
    nxy = int(round(2 * xy / 0.25)) + 1
    nt = int(round(2 * tw / 0.03)) + 1
    bx, by, bt, _ = _search(fields, x, y, th, a, r, hit, xy, tw, nxy, nt)
    bx, by, bt, s = _search(fields, bx, by, bt, a, r, hit, 0.15, 0.03, 5, 5)
    return {"ok": True, "pose": np.array([bx, by, bt]), "score": float(np.clip(s, 0, 1))}
