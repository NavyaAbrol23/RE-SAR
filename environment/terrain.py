"""Terrain and hazard type definitions shared by the ground-truth world and the robot's map."""
from __future__ import annotations

import numpy as np

# Terrain classes (ground truth)
FLOOR = 0
RUBBLE = 1
WATER = 2
BURNING = 3

TERRAIN_NAMES = {FLOOR: "floor", RUBBLE: "rubble", WATER: "water", BURNING: "burning"}

# Base traversal cost multiplier per terrain class
TERRAIN_COST = {FLOOR: 1.0, RUBBLE: 2.5, WATER: 2.0, BURNING: 3.0}

# How slippery a terrain is (multiplies the disaster's wheel_slip probability)
TERRAIN_SLIP = {FLOOR: 0.15, RUBBLE: 1.0, WATER: 1.0, BURNING: 0.3}

# Rendering colours (RGB) used by the synthetic camera and the GUI
TERRAIN_RGB = {
    FLOOR: (150, 145, 135),
    RUBBLE: (120, 95, 70),
    WATER: (60, 100, 170),
    BURNING: (200, 80, 30),
}
OBSTACLE_RGB = (70, 70, 75)


def terrain_cost_grid(terrain: np.ndarray) -> np.ndarray:
    lut = np.array([TERRAIN_COST[i] for i in range(4)], dtype=np.float32)
    return lut[terrain]


def slip_grid(terrain: np.ndarray) -> np.ndarray:
    lut = np.array([TERRAIN_SLIP[i] for i in range(4)], dtype=np.float32)
    return lut[terrain]
