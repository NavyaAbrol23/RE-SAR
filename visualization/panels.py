"""Pygame drawing helpers: buttons, text, world view, map layers, sensor previews."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pygame

from environment.terrain import OBSTACLE_RGB, TERRAIN_RGB

BG = (22, 24, 29)
PANEL = (34, 37, 44)
PANEL_EDGE = (58, 62, 72)
TEXT = (226, 229, 234)
MUTED = (150, 156, 168)
ACCENT = (90, 170, 255)
GOOD, WARN, BAD, ORANGE = (90, 200, 120), (235, 200, 70), (230, 80, 80), (245, 150, 50)
LEVEL_COLORS = {"normal": GOOD, "mild": WARN, "severe": ORANGE, "failed": BAD}
HEALTH_COLORS = {"OK": GOOD, "DEGRADED": WARN, "FAILED": BAD}
PATH_C, TARGET_C, FRONTIER_C = (230, 70, 230), (255, 90, 255), (255, 165, 40)
TRUE_C, EST_C = (255, 255, 255), (60, 220, 255)

_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        f = pygame.font.Font(None, size)
        f.set_bold(bold)
        _fonts[key] = f
    return _fonts[key]


def text(surf, s, pos, size=20, color=TEXT, bold=False, center=False):
    img = font(size, bold).render(str(s), True, color)
    r = img.get_rect()
    if center:
        r.center = pos
    else:
        r.topleft = pos
    surf.blit(img, r)
    return r


def panel(surf, rect, title=None):
    pygame.draw.rect(surf, PANEL, rect, border_radius=6)
    pygame.draw.rect(surf, PANEL_EDGE, rect, 1, border_radius=6)
    if title:
        text(surf, title, (rect[0] + 8, rect[1] + 6), 20, ACCENT, bold=True)


class Button:
    def __init__(self, rect, label, callback, color=(52, 57, 68), tooltip=""):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.color = color
        self.tooltip = tooltip

    def draw(self, surf, mouse=None):
        lab = self.label() if callable(self.label) else self.label
        col = self.color() if callable(self.color) else self.color
        hover = mouse is not None and self.rect.collidepoint(mouse)
        c = tuple(min(255, v + 25) for v in col) if hover else col
        pygame.draw.rect(surf, c, self.rect, border_radius=5)
        pygame.draw.rect(surf, PANEL_EDGE, self.rect, 1, border_radius=5)
        text(surf, lab, self.rect.center, 19, TEXT, center=True)

    def handle(self, pos):
        if self.rect.collidepoint(pos):
            self.callback()
            return True
        return False


def array_surface(rgb: np.ndarray, size) -> pygame.Surface:
    s = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
    return pygame.transform.scale(s, size)


def colormap(layer, vmin, vmax, cmap=cv2.COLORMAP_VIRIDIS):
    v = np.clip((np.nan_to_num(layer) - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    bgr = cv2.applyColorMap((v * 255).astype(np.uint8), cmap)
    return bgr[..., ::-1]


# ----------------------------------------------------------------- world view
def ground_truth_image(world) -> np.ndarray:
    lut = np.array([TERRAIN_RGB[i] for i in range(4)], np.float32)
    img = lut[world.terrain] * 0.85
    heat = world.heat[..., None]
    img = img * (1 - 0.5 * heat) + np.array([230, 70, 30]) * 0.5 * heat
    water = world.water[..., None]
    img = img * (1 - 0.4 * water) + np.array([40, 80, 200]) * 0.4 * water
    img[world.impassable] = (150, 30, 30)
    img[world.grid] = OBSTACLE_RGB
    return np.clip(img, 0, 255).astype(np.uint8)


def estimate_image(shared) -> np.ndarray:
    p = shared.occ.prob
    known = shared.occ.known
    v = np.where(known, 235 * (1 - p) + 20, 72)
    img = np.repeat(v[..., None], 3, axis=2).astype(np.float32)
    hz = shared.hazard[..., None]
    img = img * (1 - 0.6 * hz) + np.array([240, 90, 30]) * 0.6 * hz
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_robot(surf, x, y, th, cs, origin, color, filled=True):
    ox, oy = origin
    cx, cy = ox + x * cs, oy + y * cs
    r = cs * 0.55
    pts = [(cx + r * math.cos(th), cy + r * math.sin(th)),
           (cx + r * 0.7 * math.cos(th + 2.5), cy + r * 0.7 * math.sin(th + 2.5)),
           (cx + r * 0.7 * math.cos(th - 2.5), cy + r * 0.7 * math.sin(th - 2.5))]
    pygame.draw.polygon(surf, color, pts, 0 if filled else 2)


def draw_world(surf, mission, origin, cs, show_truth=True, show_lidar=True, gt_cache=None):
    w = mission.world
    ox, oy = origin
    base = gt_cache if (show_truth and gt_cache is not None) else (ground_truth_image(w) if show_truth else estimate_image(mission.shared))
    surf.blit(array_surface(base, (w.W * cs, w.H * cs)), origin)
    # frontiers
    for i, f in enumerate(mission.frontiers):
        r = pygame.Rect(ox + f.x * cs + cs // 4, oy + f.y * cs + cs // 4, cs // 2, cs // 2)
        pygame.draw.rect(surf, FRONTIER_C, r, 0 if i == mission.target_index and mission.target else 1)
    # A* path + target
    if mission.target is not None and mission.path:
        x, y, _ = mission.ekf.pose
        pts = [(ox + x * cs, oy + y * cs)] + [(ox + (p[0] + 0.5) * cs, oy + (p[1] + 0.5) * cs) for p in mission.path]
        if len(pts) > 1:
            pygame.draw.lines(surf, PATH_C, False, pts, 2)
    if mission.target is not None:
        tx, ty = ox + (mission.target.x + 0.5) * cs, oy + (mission.target.y + 0.5) * cs
        d = cs * 0.45
        pygame.draw.line(surf, TARGET_C, (tx - d, ty - d), (tx + d, ty + d), 3)
        pygame.draw.line(surf, TARGET_C, (tx - d, ty + d), (tx + d, ty - d), 3)
    # trajectories
    if len(mission.true_traj) > 1:
        pygame.draw.lines(surf, TRUE_C, False, [(ox + p[0] * cs, oy + p[1] * cs) for p in mission.true_traj[-600:]], 1)
        pygame.draw.lines(surf, EST_C, False, [(ox + p[0] * cs, oy + p[1] * cs) for p in mission.est_traj[-600:]], 1)
    # LiDAR endpoints (estimated frame)
    obs = mission.last.get("obs")
    if show_lidar and obs is not None:
        x, y, th = mission.ekf.pose
        d = obs["lidar"].data
        ok = np.isfinite(d["ranges"])
        for a, r in zip(d["angles"][ok], d["ranges"][ok]):
            px, py = x + r * math.cos(th + a), y + r * math.sin(th + a)
            pygame.draw.circle(surf, (110, 255, 140), (int(ox + px * cs), int(oy + py * cs)), 2)
    # ground-truth victims: white rings with a "GT" tag
    for v in w.victims:
        c = (int(ox + v.x * cs), int(oy + v.y * cs))
        pygame.draw.circle(surf, (255, 255, 255), c, int(cs * 0.55), 2)
        text(surf, "GT", (c[0] - 7, c[1] - cs - 2), 15, (255, 255, 255))
    # robot's victim tracks: yellow (tentative) / green (confirmed)
    for t in mission.tracker.tracks:
        c = (int(ox + t.x * cs), int(oy + t.y * cs))
        if t.confirmed:
            pygame.draw.circle(surf, GOOD, c, int(cs * 0.38))
            text(surf, f"V{t.tid}", (c[0] + 6, c[1] + 4), 15, GOOD)
        elif t.prob > 0.3:
            pygame.draw.circle(surf, WARN, c, int(cs * 0.3), 2)
    # robot: true pose (white) and estimated pose (cyan outline)
    draw_robot(surf, w.x, w.y, w.theta, cs, origin, TRUE_C)
    x, y, th = mission.ekf.pose
    draw_robot(surf, x, y, th, cs, origin, EST_C, filled=False)


# --------------------------------------------------------------- sensor previews
def rgb_preview(img, size):
    return array_surface(img, size)


def thermal_preview(T, size):
    return array_surface(colormap(T, 10, 80, cv2.COLORMAP_INFERNO), size)


def lidar_preview(surf, rect, data):
    pygame.draw.rect(surf, (15, 17, 20), rect)
    cx, cy = rect.center
    scale = (rect.width / 2 - 4) / data["max_range"]
    for a, r in zip(data["angles"], data["ranges"]):
        ang = a - math.pi / 2  # robot forward = up
        if np.isnan(r):
            pygame.draw.line(surf, (110, 40, 40), (cx, cy), (cx + 6 * math.cos(ang), cy + 6 * math.sin(ang)), 1)
            continue
        rr = data["max_range"] if np.isinf(r) else r
        col = (60, 90, 70) if np.isinf(r) else (110, 255, 140)
        pygame.draw.circle(surf, col, (int(cx + rr * scale * math.cos(ang)), int(cy + rr * scale * math.sin(ang))), 2)
    pygame.draw.circle(surf, EST_C, (cx, cy), 3)


def bar(surf, rect, value, color, marker=None):
    pygame.draw.rect(surf, (55, 59, 68), rect, border_radius=3)
    w = int(rect[2] * float(np.clip(value, 0, 1)))
    if w > 0:
        pygame.draw.rect(surf, color, (rect[0], rect[1], w, rect[3]), border_radius=3)
    if marker is not None:
        mx = rect[0] + int(rect[2] * float(np.clip(marker, 0, 1)))
        pygame.draw.line(surf, (255, 255, 255), (mx, rect[1] - 2), (mx, rect[1] + rect[3] + 2), 2)
