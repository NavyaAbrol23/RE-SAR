"""Thermal anomaly detector: compact blobs in the human temperature band that stand out from
their local background. Hot environmental sources (> 40 C) and large warm regions are rejected.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_thermal(T: np.ndarray, t_sat: float, geo, band=(30.0, 40.0)) -> list[dict]:
    fov = geo.fov_mask
    ppc = geo.ppc
    m = (T >= band[0]) & (T <= band[1]) & (T < t_sat - 0.5) & fov
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < 0.15 * ppc * ppc or area > 1.2 * ppc * ppc:
            continue
        fill = area / max(w * h, 1)
        if fill < 0.35:
            continue  # thin rings around fires etc.
        blob = lab == i
        ring = cv2.dilate(blob.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~blob & fov
        if ring.sum() < 4:
            continue
        contrast = float(T[blob].mean() - np.median(T[ring]))
        if contrast < 3.0:
            continue
        conf = float(np.clip(0.35 + contrast / 25.0 + 0.2 * fill, 0.1, 0.95))
        col, row = cents[i]
        f, l = geo.pixel_to_local(row, col)
        out.append({"f": f, "l": l, "conf": conf, "sigma": 0.5, "modality": "thermal"})
    return out
