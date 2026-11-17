"""Synthetic RGB camera: egocentric local view with smoke, dust, darkness, blur and noise."""
from __future__ import annotations

import math

import cv2
import numpy as np

from environment.terrain import OBSTACLE_RGB, TERRAIN_RGB
from sensors.base_sensor import BaseSensor, SensorObservation
from sensors.camera_common import EgoGeometry, cell_lookup

SKIN = (205, 160, 130)
CLOTHES = ((40, 60, 150), (120, 40, 140), (60, 120, 60), (40, 140, 150))  # blue, purple, green, teal


class RGBCamera(BaseSensor):
    name = "rgb"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        c = cfg["sensors"]["rgb"]
        self.geo = EgoGeometry(c["range"], c["px_per_cell"], c["fov_deg"])
        self.terrain_lut = np.array([TERRAIN_RGB[i] for i in range(4)], dtype=np.float32)

    def render_clean(self, world):
        g = self.geo
        wx, wy = g.world_coords(world.x, world.y, world.theta)
        cx, cy, oob = cell_lookup(world, wx, wy)
        img = self.terrain_lut[world.terrain[cy, cx]].copy()
        vis = world.visible_mask(world.x, world.y, world.theta, math.radians(self.cfg["sensors"]["rgb"]["fov_deg"]) + 0.3, g.R + 1)
        occluded = ~vis[cy, cx] | world.grid[cy, cx] | oob
        img[occluded] = OBSTACLE_RGB
        ppc = g.ppc
        for v in world.victims:
            vcx, vcy = int(v.x), int(v.y)
            if not vis[vcy, vcx]:
                continue
            r, c = g.to_pixel(world.x, world.y, world.theta, v.x, v.y)
            if not (0 <= r < g.size and 0 <= c < g.size):
                continue
            ang = (v.vid * 47) % 180
            cv2.ellipse(img, (int(c), int(r)), (int(ppc * 0.55), int(ppc * 0.32)), ang, 0, 360,
                        CLOTHES[v.vid % len(CLOTHES)], -1)
            hx = c + 0.6 * ppc * math.cos(math.radians(ang))
            hy = r + 0.6 * ppc * math.sin(math.radians(ang))
            cv2.circle(img, (int(round(hx)), int(round(hy))), max(2, int(ppc * 0.28)), SKIN, -1)  # head
        for d in world.decoys:
            if d.kind != "visual" or not vis[int(d.y), int(d.x)]:
                continue
            r, c = g.to_pixel(world.x, world.y, world.theta, d.x, d.y)
            if 0 <= r < g.size and 0 <= c < g.size:
                cv2.circle(img, (int(c), int(r)), max(1, int(ppc * 0.3)), (195, 150, 115), -1)
        img[~g.fov_mask] = 0
        return img

    def observe(self, world, cond, manual, failed, t, ctx):
        g = self.geo
        if failed:  # black frame from a dead camera
            return SensorObservation(self.name, t, True, {"image": np.zeros((g.size, g.size, 3), np.uint8),
                                                          "fov_mask": g.fov_mask})
        img = self.render_clean(world)
        haze = np.clip(0.9 * cond["smoke"] + 0.5 * cond["dust"] + 0.6 * manual, 0, 0.97)
        img = img * (1 - haze) + 165.0 * haze
        sigma = 0.3 + 2.5 * cond["dust"] + 1.5 * cond["smoke"] + 2.0 * manual
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        img *= (1 - cond["darkness"]) * (1 - 0.5 * manual)
        img += self.rng.normal(0, 2 + 25 * manual + 6 * cond["darkness"], img.shape)
        img[~g.fov_mask] = 0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return SensorObservation(self.name, t, True, {"image": img, "fov_mask": g.fov_mask})
