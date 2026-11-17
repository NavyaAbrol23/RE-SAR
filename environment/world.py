"""Ground-truth disaster world. Only sensors, rendering and evaluation may read this state."""
from __future__ import annotations

import math

import numpy as np

from environment.hazard_generator import generate_hazards
from environment.map_generator import generate_layout, reachable_mask
from environment.terrain import slip_grid, terrain_cost_grid
from environment.victim_generator import place_decoys, place_victims


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class World:
    def __init__(self, cfg: dict, seed: int | None = None):
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 0) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        w = cfg["world"]
        self.disaster = cfg["disaster"]
        self.severity = float(np.clip(w["severity"], 0.0, 1.0))
        self.difficulty = w["difficulty"]
        self.env = cfg.get("environment", {})
        self.W, self.H = int(w["width"]), int(w["height"])

        self.grid, rubble, start = generate_layout(self.W, self.H, self.disaster, self.difficulty,
                                                   self.severity, self.env, self.rng)
        f = generate_hazards(self.grid, rubble, start, self.disaster, self.severity, self.env, self.rng)
        self.terrain, self.hazard = f["terrain"], f["hazard"]
        self.heat, self.water, self.smoke, self.dust = f["heat"], f["water"], f["smoke"], f["dust"]
        self.impassable = (self.hazard >= w["impassable_hazard"]) & ~self.grid
        self.terrain_cost = terrain_cost_grid(self.terrain)
        self.slip = slip_grid(self.terrain)

        self.darkness = float(np.clip(self.env.get("darkness", 0.0) * (0.5 + 0.5 * self.severity), 0, 0.95))
        self.acoustic_noise = float(np.clip(self.env.get("acoustic_noise", 0.3) * (0.5 + 0.7 * self.severity), 0, 1))

        self.start = (start[0] + 0.5, start[1] + 0.5)
        self.reachable = reachable_mask(self.grid | self.impassable, start)
        a = cfg["sensors"]["audio"]
        self.victims = place_victims(self.reachable, start, int(w["num_victims"]), self.rng,
                                     a["responsive_fraction"])
        self.decoys = place_decoys(self.reachable, self.victims, int(w["num_visual_decoys"]),
                                   int(w["num_thermal_decoys"]), self.rng, self.disaster)
        self.reset_robot()

    # ------------------------------------------------------------------ robot
    def reset_robot(self):
        self.x, self.y = self.start
        self.theta = 0.0
        self.collisions = 0
        self.hazard_exposure = 0.0
        self.trajectory = [(self.x, self.y, self.theta)]

    @property
    def pose(self):
        return np.array([self.x, self.y, self.theta], dtype=np.float64)

    def cell_of(self, x, y):
        return int(math.floor(x)), int(math.floor(y))

    def in_bounds(self, cx, cy):
        return 0 <= cx < self.W and 0 <= cy < self.H

    def blocked(self, x, y):
        cx, cy = self.cell_of(x, y)
        return (not self.in_bounds(cx, cy)) or bool(self.grid[cy, cx])

    def apply_motion(self, d_theta_cmd: float, dist_cmd: float, rng: np.random.Generator) -> dict:
        """Execute a rotate-then-drive command on the TRUE robot. Returns ground-truth outcome."""
        r = self.cfg["robot"]
        cx, cy = self.cell_of(self.x, self.y)
        slip_p = float(self.env.get("wheel_slip", 0.1) * (0.5 + self.severity) * self.slip[cy, cx])
        rot_noise = r["rot_noise"] * (1 + 3 * slip_p)
        actual_rot = d_theta_cmd + rng.normal(0, rot_noise)
        self.theta = wrap(self.theta + actual_rot)
        slipped = bool(rng.random() < slip_p) and dist_cmd > 0
        if slipped:
            actual_dist = dist_cmd * rng.uniform(0.0, 0.4)
        else:
            actual_dist = max(0.0, dist_cmd + rng.normal(0, r["move_noise"]) * (dist_cmd > 0))
        nx = self.x + actual_dist * math.cos(self.theta)
        ny = self.y + actual_dist * math.sin(self.theta)
        mx, my = (self.x + nx) / 2, (self.y + ny) / 2
        collided = False
        if self.blocked(nx, ny) or self.blocked(mx, my):
            collided = True
            self.collisions += 1
            actual_dist = 0.0
        else:
            self.x, self.y = nx, ny
        ncx, ncy = self.cell_of(self.x, self.y)
        exposure = float(self.hazard[ncy, ncx])
        self.hazard_exposure += exposure
        self.trajectory.append((self.x, self.y, self.theta))
        return {"actual_rot": actual_rot, "actual_dist": actual_dist, "slipped": slipped,
                "collided": collided, "exposure": exposure, "slip_p": slip_p}

    # -------------------------------------------------------------- geometry
    def cast_rays(self, x, y, angles, max_range, step=0.1):
        """Vectorised ray marching. Returns ranges (inf where nothing hit within max_range)."""
        angles = np.asarray(angles, dtype=np.float64)
        ts = np.arange(step, max_range + 1e-9, step)
        px = x + np.cos(angles)[:, None] * ts[None, :]
        py = y + np.sin(angles)[:, None] * ts[None, :]
        cx, cy = np.floor(px).astype(int), np.floor(py).astype(int)
        oob = (cx < 0) | (cy < 0) | (cx >= self.W) | (cy >= self.H)
        occ = np.ones_like(oob)
        ok = ~oob
        occ[ok] = self.grid[cy[ok], cx[ok]]
        hit_any = occ.any(axis=1)
        first = occ.argmax(axis=1)
        ranges = np.where(hit_any, ts[first], np.inf)
        return ranges

    def visible_mask(self, x, y, theta, fov, rng_cells, n_rays=90, step=0.25):
        """Cells visible from (x, y) within a field of view (line of sight respected)."""
        angles = theta + np.linspace(-fov / 2, fov / 2, n_rays)
        ts = np.arange(0.0, rng_cells + 1e-9, step)
        px = x + np.cos(angles)[:, None] * ts[None, :]
        py = y + np.sin(angles)[:, None] * ts[None, :]
        cx, cy = np.floor(px).astype(int), np.floor(py).astype(int)
        cx = np.clip(cx, 0, self.W - 1)
        cy = np.clip(cy, 0, self.H - 1)
        occ = self.grid[cy, cx]
        # keep samples up to and including the first occupied sample
        blocked_before = np.cumsum(occ, axis=1) - occ > 0
        keep = ~blocked_before
        mask = np.zeros((self.H, self.W), dtype=bool)
        mask[cy[keep], cx[keep]] = True
        return mask

    def walls_between(self, x0, y0, x1, y1, step=0.3):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(d / step))
        ts = np.linspace(0, 1, n)
        cx = np.floor(x0 + (x1 - x0) * ts).astype(int)
        cy = np.floor(y0 + (y1 - y0) * ts).astype(int)
        occ = self.grid[np.clip(cy, 0, self.H - 1), np.clip(cx, 0, self.W - 1)]
        # count transitions into occupied runs
        return int(np.sum(np.diff(occ.astype(int)) == 1))

    # ------------------------------------------------------------- metrics
    def ground_truth_conditions(self):
        """Local environmental conditions at the true robot cell (used by sensor generators)."""
        cx, cy = self.cell_of(self.x, self.y)
        return {
            "smoke": float(self.smoke[cy, cx]),
            "dust": float(self.dust[cy, cx]),
            "heat": float(self.heat[max(0, cy - 3):cy + 4, max(0, cx - 3):cx + 4].max()),
            "water": float(self.water[cy, cx]),
            "darkness": self.darkness,
            "acoustic_noise": self.acoustic_noise,
            "slip": float(self.env.get("wheel_slip", 0.1) * (0.5 + self.severity) * self.slip[cy, cx]),
        }
