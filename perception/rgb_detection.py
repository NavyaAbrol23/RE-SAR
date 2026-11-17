"""Image-processing victim-candidate detector for the RGB camera.

1. Auto-gain (95th percentile of the field of view -> 200) to compensate for darkness.
2. HSV skin-tone mask, morphological cleanup, connected components.
3. Size filter; confidence rises if saturated non-skin "clothing" pixels surround the blob
   (debris with a skin-like colour has no clothing ring and gets lower confidence).
Outputs candidates in robot-local coordinates (forward f, right l) in cells.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_rgb(img: np.ndarray, geo) -> list[dict]:
    fov = geo.fov_mask
    px = img[fov]
    if px.size == 0 or px.mean() < 3:
        return []
    gain = 200.0 / max(float(np.percentile(px, 95)), 10.0)
    im = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    skin = (h <= 17) & (s >= 35) & (s <= 150) & (v >= 150) & fov
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(skin, 8)
    ppc = geo.ppc
    out = []
    clothes = (s > 55) & (v > 40) & (h > 25)  # saturated non-skin, non-fire hues
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2 or area > 0.8 * ppc * ppc:
            continue
        blob = (lab == i).astype(np.uint8)
        ring = cv2.dilate(blob, np.ones((5, 5), np.uint8)) & (1 - blob)
        rc = float(clothes[ring.astype(bool)].mean()) if ring.any() else 0.0
        if rc < 0.12:
            continue  # skin-coloured debris / rubble: no clothing next to it
        conf = float(np.clip(0.35 + 1.2 * rc, 0.1, 0.95))
        col, row = cents[i]
        f, l = geo.pixel_to_local(row, col)
        out.append({"f": f, "l": l, "conf": conf, "sigma": 0.6, "modality": "rgb"})
    # merge blobs of one person (head + hands) -> keep the most confident per 1-cell radius
    out.sort(key=lambda c: -c["conf"])
    merged = []
    for c in out:
        if all(np.hypot(c["f"] - m["f"], c["l"] - m["l"]) > 1.0 for m in merged):
            merged.append(c)
    return merged
