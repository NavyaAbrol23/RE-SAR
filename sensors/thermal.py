"""Synthetic thermal camera (temperatures in deg C) with heat interference, noise and saturation."""
from __future__ import annotations

import math

import cv2
import numpy as np

from sensors.base_sensor import BaseSensor, SensorObservation
from sensors.camera_common import EgoGeometry, cell_lookup
from environment.terrain import WATER


class ThermalCamera(BaseSensor):
    name = "thermal"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        c = cfg["sensors"]["thermal"]
        self.geo = EgoGeometry(c["range"], c["px_per_cell"], c["fov_deg"])
        self.t_sat = float(c["t_sat"])

    def observe(self, world, cond, manual, failed, t, ctx):
        g = self.geo
        heat_if = cond["heat"] * world.env.get("heat", 0.0)
        t_sat = max(25.0, self.t_sat * (1 - 0.7 * manual) * (1 - 0.6 * heat_if))
        if failed:  # fully saturated sensor
            img = np.full((g.size, g.size), t_sat, np.float32)
            img[~g.fov_mask] = 0
            return SensorObservation(self.name, t, True, {"temp": img, "t_sat": t_sat, "fov_mask": g.fov_mask})
        wx, wy = g.world_coords(world.x, world.y, world.theta)
        cx, cy, oob = cell_lookup(world, wx, wy)
        T = 20.0 + 130.0 * world.heat[cy, cx] + 15.0 * world.smoke[cy, cx]
        T = np.where(world.terrain[cy, cx] == WATER, 12.0, T).astype(np.float32)
        vis = world.visible_mask(world.x, world.y, world.theta, math.radians(self.cfg["sensors"]["thermal"]["fov_deg"]) + 0.3, g.R + 1)
        ppc = g.ppc
        for v in world.victims:
            if not vis[int(v.y), int(v.x)]:
                continue
            r, c = g.to_pixel(world.x, world.y, world.theta, v.x, v.y)
            if 0 <= r < g.size and 0 <= c < g.size:
                ang = (v.vid * 47) % 180
                cv2.ellipse(T, (int(c), int(r)), (int(ppc * 0.5), int(ppc * 0.3)), ang, 0, 360,
                            float(v.temperature - 1.5), -1)
                hx = c + 0.5 * ppc * math.cos(math.radians(ang))
                hy = r + 0.5 * ppc * math.sin(math.radians(ang))
                cv2.circle(T, (int(hx), int(hy)), max(1, int(ppc * 0.22)), float(v.temperature), -1)
        for d in world.decoys:
            if d.kind != "thermal" or not vis[int(d.y), int(d.x)]:
                continue
            r, c = g.to_pixel(world.x, world.y, world.theta, d.x, d.y)
            if 0 <= r < g.size and 0 <= c < g.size:
                h = int(ppc * 0.4)
                cv2.rectangle(T, (int(c) - h, int(r) - h), (int(c) + h, int(r) + h), float(d.temperature), -1)
        T = cv2.GaussianBlur(T, (0, 0), 0.8 + 0.8 * cond["smoke"] + 1.5 * manual)
        T += self.rng.normal(0, 0.4 + 4 * manual + 2 * heat_if, T.shape).astype(np.float32)
        T = np.clip(T, 0, t_sat)
        T[~g.fov_mask] = 0
        return SensorObservation(self.name, t, True, {"temp": T.astype(np.float32), "t_sat": t_sat,
                                                      "fov_mask": g.fov_mask})
