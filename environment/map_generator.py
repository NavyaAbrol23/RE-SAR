"""Procedural building layouts with disaster-specific structural damage."""
from __future__ import annotations

from collections import deque

import numpy as np

DIFFICULTY = {
    "easy": {"min_room": 9, "door": 3, "clutter": 0.02},
    "medium": {"min_room": 7, "door": 2, "clutter": 0.04},
    "hard": {"min_room": 5, "door": 2, "clutter": 0.07},
}


def _divide(grid, x0, y0, x1, y1, rng, min_room, door):
    """Recursive division: split a room with a wall containing a door."""
    w, h = x1 - x0, y1 - y0
    if w < 2 * min_room and h < 2 * min_room:
        return
    vertical = w > h if w != h else rng.random() < 0.5
    if vertical and w >= 2 * min_room:
        wx = int(rng.integers(x0 + min_room, x1 - min_room + 1))
        grid[y0:y1, wx] = True
        dy = int(rng.integers(y0, max(y0 + 1, y1 - door)))
        grid[dy:dy + door, wx] = False
        _divide(grid, x0, y0, wx, y1, rng, min_room, door)
        _divide(grid, wx + 1, y0, x1, y1, rng, min_room, door)
    elif h >= 2 * min_room:
        wy = int(rng.integers(y0 + min_room, y1 - min_room + 1))
        grid[wy, x0:x1] = True
        dx = int(rng.integers(x0, max(x0 + 1, x1 - door)))
        grid[wy, dx:dx + door] = False
        _divide(grid, x0, y0, x1, wy, rng, min_room, door)
        _divide(grid, x0, wy + 1, x1, y1, rng, min_room, door)


def _blob(grid, cx, cy, r, value, rng):
    H, W = grid.shape
    for y in range(max(1, cy - r), min(H - 1, cy + r + 1)):
        for x in range(max(1, cx - r), min(W - 1, cx + r + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r + rng.integers(-1, 2):
                grid[y, x] = value


def reachable_mask(blocked: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """4-connected flood fill over non-blocked cells from start (x, y)."""
    H, W = blocked.shape
    seen = np.zeros_like(blocked, dtype=bool)
    sx, sy = start
    if blocked[sy, sx]:
        return seen
    q = deque([(sx, sy)])
    seen[sy, sx] = True
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not seen[ny, nx] and not blocked[ny, nx]:
                seen[ny, nx] = True
                q.append((nx, ny))
    return seen


def generate_layout(width: int, height: int, disaster: str, difficulty: str, severity: float,
                    env_cfg: dict, rng: np.random.Generator):
    """Return (obstacle grid bool[H,W], rubble mask bool[H,W], start (x, y))."""
    p = DIFFICULTY.get(difficulty, DIFFICULTY["medium"])
    grid = np.zeros((height, width), dtype=bool)
    grid[0, :] = grid[-1, :] = True
    grid[:, 0] = grid[:, -1] = True
    _divide(grid, 1, 1, width - 1, height - 1, rng, p["min_room"], p["door"])

    rubble = np.zeros_like(grid)
    if disaster == "earthquake":
        # Collapsed wall segments open new gaps, rubble piles block passages
        wall_cells = np.argwhere(grid[1:-1, 1:-1]) + 1
        n_break = int(len(wall_cells) * 0.08)
        for y, x in wall_cells[rng.choice(len(wall_cells), size=min(n_break, len(wall_cells)), replace=False)]:
            grid[y, x] = False
        n_piles = int(4 + 8 * severity * (1 + p["clutter"] * 10))
        for _ in range(n_piles):
            cx, cy = int(rng.integers(2, width - 2)), int(rng.integers(2, height - 2))
            _blob(grid, cx, cy, int(rng.integers(1, 3)), True, rng)
        n_rub = int(env_cfg.get("rubble_density", 0.1) * width * height / 6 * (0.5 + severity))
        for _ in range(n_rub):
            cx, cy = int(rng.integers(1, width - 1)), int(rng.integers(1, height - 1))
            _blob(rubble, cx, cy, int(rng.integers(1, 3)), True, rng)
    elif disaster == "flood":
        # Water damage: many interior walls washed out, floating debris
        wall_cells = np.argwhere(grid[1:-1, 1:-1]) + 1
        n_break = int(len(wall_cells) * 0.25)
        for y, x in wall_cells[rng.choice(len(wall_cells), size=min(n_break, len(wall_cells)), replace=False)]:
            grid[y, x] = False
    # Generic clutter (furniture / debris)
    clutter = rng.random(grid.shape) < p["clutter"] * (0.6 + 0.8 * severity)
    clutter[[0, -1], :] = False
    clutter[:, [0, -1]] = False
    grid |= clutter
    if disaster == "flood" or disaster == "fire":
        n_rub = int(env_cfg.get("rubble_density", 0.03) * width * height / 8)
        for _ in range(n_rub):
            cx, cy = int(rng.integers(1, width - 1)), int(rng.integers(1, height - 1))
            _blob(rubble, cx, cy, 1, True, rng)

    # Start position: clear 3x3 area in the left third
    for _ in range(200):
        sx, sy = int(rng.integers(2, max(3, width // 4))), int(rng.integers(2, height - 2))
        if not grid[sy - 1:sy + 2, sx - 1:sx + 2].any():
            break
    else:
        sx, sy = 2, 2
    grid[sy - 1:sy + 2, sx - 1:sx + 2] = False
    rubble[sy - 1:sy + 2, sx - 1:sx + 2] = False

    reach = reachable_mask(grid, (sx, sy))
    grid[~reach] = True  # seal unreachable pockets so the map is fully explorable
    rubble &= ~grid
    return grid, rubble, (sx, sy)
