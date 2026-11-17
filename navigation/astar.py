"""8-connected A* (and Dijkstra) on a costmap. No diagonal corner cutting.
Edge cost = step_length * cost(destination cell). Heuristic = octile distance * min cell cost.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def _neighbors(cost, x, y):
    H, W = cost.shape
    for dx, dy, l in MOVES:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H) or not np.isfinite(cost[ny, nx]):
            continue
        if dx and dy and (not np.isfinite(cost[y, nx]) or not np.isfinite(cost[ny, x])):
            continue
        yield nx, ny, l * cost[ny, nx]


def astar(cost: np.ndarray, start, goal):
    """start/goal (x, y) cells. Returns list of cells from start (excluded) to goal, or None."""
    H, W = cost.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= gx < W and 0 <= gy < H) or not np.isfinite(cost[gy, gx]):
        return None
    if (sx, sy) == (gx, gy):
        return []
    cmin = float(np.min(cost[np.isfinite(cost)])) if np.isfinite(cost).any() else 1.0

    def h(x, y):
        dx, dy = abs(x - gx), abs(y - gy)
        return cmin * (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy))

    g = {(sx, sy): 0.0}
    parent = {}
    openq = [(h(sx, sy), 0.0, sx, sy)]
    closed = set()
    while openq:
        _, gc, x, y = heapq.heappop(openq)
        if (x, y) in closed:
            continue
        if (x, y) == (gx, gy):
            path = [(x, y)]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path.reverse()
            return path[1:]
        closed.add((x, y))
        for nx, ny, c in _neighbors(cost, x, y):
            ng = gc + c
            if ng < g.get((nx, ny), math.inf):
                g[(nx, ny)] = ng
                parent[(nx, ny)] = (x, y)
                heapq.heappush(openq, (ng + h(nx, ny), ng, nx, ny))
    return None


def dijkstra(cost: np.ndarray, start) -> np.ndarray:
    """Cost-to-reach from start for every cell (inf if unreachable)."""
    H, W = cost.shape
    dist = np.full((H, W), np.inf)
    sx, sy = start
    dist[sy, sx] = 0.0
    q = [(0.0, sx, sy)]
    while q:
        d, x, y = heapq.heappop(q)
        if d > dist[y, x]:
            continue
        for nx, ny, c in _neighbors(cost, x, y):
            nd = d + c
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(q, (nd, nx, ny))
    return dist


def path_length(path, start) -> float:
    pts = [start] + list(path)
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])))
