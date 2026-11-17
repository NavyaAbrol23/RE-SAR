"""Victim and decoy placement (ground truth only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Victim:
    vid: int
    x: float          # continuous cell coordinates (cell centre = i + 0.5)
    y: float
    temperature: float
    responsive: bool  # emits distress calls


@dataclass
class Decoy:
    kind: str         # "visual" (skin-coloured debris) or "thermal" (warm object)
    x: float
    y: float
    temperature: float


def _pick_cells(mask, n, rng, min_sep, avoid=()):
    cand = np.argwhere(mask)
    rng.shuffle(cand)
    chosen = list(avoid)
    out = []
    for cy, cx in cand:
        if all((cx - ax) ** 2 + (cy - ay) ** 2 >= min_sep ** 2 for ax, ay in chosen):
            chosen.append((cx, cy))
            out.append((cx, cy))
            if len(out) == n:
                break
    return out


def place_victims(free_reachable: np.ndarray, start, n: int, rng: np.random.Generator,
                  responsive_fraction: float = 0.75):
    H, W = free_reachable.shape
    yy, xx = np.mgrid[0:H, 0:W]
    far = free_reachable & (np.hypot(xx - start[0], yy - start[1]) >= 6)
    cells = _pick_cells(far, n, rng, min_sep=4)
    if len(cells) < n:  # relax separation on tiny maps
        cells += _pick_cells(far, n - len(cells), rng, min_sep=2, avoid=cells)
    return [Victim(i, cx + 0.5, cy + 0.5, float(rng.uniform(33.5, 37.0)),
                   bool(rng.random() < responsive_fraction)) for i, (cx, cy) in enumerate(cells)]


def place_decoys(free_reachable: np.ndarray, victims, n_visual: int, n_thermal: int,
                 rng: np.random.Generator, disaster: str):
    avoid = [(int(v.x), int(v.y)) for v in victims]
    decoys = []
    for cx, cy in _pick_cells(free_reachable, n_visual, rng, 3, avoid):
        decoys.append(Decoy("visual", cx + 0.5, cy + 0.5, 20.0))
        avoid.append((cx, cy))
    n_th = n_thermal + (2 if disaster == "fire" else 0)
    for cx, cy in _pick_cells(free_reachable, n_th, rng, 3, avoid):
        decoys.append(Decoy("thermal", cx + 0.5, cy + 0.5, float(rng.uniform(36.0, 55.0))))
    return decoys
