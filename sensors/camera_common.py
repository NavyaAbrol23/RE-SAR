"""Shared egocentric camera geometry for the RGB and thermal cameras.

Images are egocentric top-down local views: image "up" = robot forward, image "right" = robot right.
Pixel (row i, col j) maps to robot-frame offsets  forward f = (c - i)/ppc,  right l = (j - c)/ppc.
World offset = f * (cos th, sin th) + l * (cos(th + pi/2), sin(th + pi/2)).
Detectors invert this with the robot's ESTIMATED heading, so localisation error propagates honestly.
"""
from __future__ import annotations

import math

import numpy as np


class EgoGeometry:
    def __init__(self, range_cells: int, ppc: int, fov_deg: float):
        self.R, self.ppc = int(range_cells), int(ppc)
        self.size = (2 * self.R + 1) * self.ppc
        self.c = (self.size - 1) / 2.0
        ii, jj = np.mgrid[0:self.size, 0:self.size].astype(np.float64)
        self.f = (self.c - ii) / self.ppc
        self.l = (jj - self.c) / self.ppc
        self.dist = np.hypot(self.f, self.l)
        bearing = np.arctan2(self.l, self.f)
        self.fov_mask = (np.abs(bearing) <= math.radians(fov_deg) / 2) & (self.dist <= self.R)
        self.fov_mask |= self.dist < 0.6  # robot's own surroundings

    def world_coords(self, x, y, th):
        wx = x + self.f * math.cos(th) + self.l * math.cos(th + math.pi / 2)
        wy = y + self.f * math.sin(th) + self.l * math.sin(th + math.pi / 2)
        return wx, wy

    def to_pixel(self, x, y, th, px, py):
        """World point -> (row, col) pixel in the egocentric image."""
        dx, dy = px - x, py - y
        f = dx * math.cos(th) + dy * math.sin(th)
        l = dx * math.cos(th + math.pi / 2) + dy * math.sin(th + math.pi / 2)
        return self.c - f * self.ppc, self.c + l * self.ppc

    def pixel_to_local(self, row, col):
        return (self.c - row) / self.ppc, (col - self.c) / self.ppc

    @staticmethod
    def local_to_world(x, y, th, f, l):
        return (x + f * math.cos(th) + l * math.cos(th + math.pi / 2),
                y + f * math.sin(th) + l * math.sin(th + math.pi / 2))


def cell_lookup(world, wx, wy):
    cx = np.clip(np.floor(wx).astype(int), 0, world.W - 1)
    cy = np.clip(np.floor(wy).astype(int), 0, world.H - 1)
    oob = (wx < 0) | (wy < 0) | (wx >= world.W) | (wy >= world.H)
    return cx, cy, oob
