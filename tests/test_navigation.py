import math

import numpy as np

from config.loader import load_config
from core.mission import Mission
from navigation.astar import astar, dijkstra, path_length
from navigation.frontier import frontier_mask
from navigation.safety import SafetyController


def test_astar_valid_path():
    cost = np.ones((10, 10))
    cost[5, :8] = np.inf
    path = astar(cost, (0, 0), (0, 9))
    assert path is not None and path[-1] == (0, 9)
    for (x, y) in path:
        assert np.isfinite(cost[y, x])
    pts = [(0, 0)] + path
    for a, b in zip(pts, pts[1:]):  # 8-connected steps
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def test_astar_no_corner_cutting_and_unreachable():
    cost = np.ones((3, 3))
    cost[0, 1] = cost[1, 0] = np.inf
    assert astar(cost, (0, 0), (1, 1)) is None
    cost = np.ones((5, 5))
    cost[:, 2] = np.inf
    assert astar(cost, (0, 0), (4, 4)) is None
    assert astar(cost, (0, 0), (2, 2)) is None  # goal inside obstacle


def test_astar_prefers_cheap_cells():
    cost = np.ones((5, 9))
    cost[2, 1:8] = 50.0  # expensive hazard band on the straight line
    path = astar(cost, (0, 2), (8, 2))
    assert path_length(path, (0, 2)) > 8 and sum(cost[y, x] for x, y in path) < 50


def test_dijkstra_matches_astar_cost():
    rng = np.random.default_rng(1)
    cost = rng.uniform(1, 3, (12, 12))
    d = dijkstra(cost, (0, 0))
    path = astar(cost, (0, 0), (11, 11))
    c = sum(math.hypot(b[0] - a[0], b[1] - a[1]) * cost[b[1], b[0]] for a, b in zip([(0, 0)] + path, path))
    assert abs(c - d[11, 11]) < 1e-6


def test_safety_controller_vetoes_occupied_and_hazard():
    m = Mission(load_config("fire"), seed=0, nav="nearest")
    sc = SafetyController(m.cfg)
    cx, cy = m.est_cell()
    m.shared.occ.L[cy, cx + 1] = 4.0
    ok, why = sc.check(m.shared, m.ekf.pose, (cx + 1, cy), None, 0.0)
    assert not ok and why == "occupied"
    m.shared.terrain.heat[cy + 1, cx] = 1.0
    ok, why = sc.check(m.shared, m.ekf.pose, (cx, cy + 1), None, 0.0)
    assert not ok and why == "hazard"


def test_frontiers_exist_at_start_and_are_reachable():
    m = Mission(load_config("earthquake"), seed=3, nav="nearest")
    fr = m.compute_frontiers()
    assert fr and all(np.isfinite(f.dist) for f in fr)
    assert [f.dist for f in fr] == sorted(f.dist for f in fr)  # ordering used by PPO actions
    assert frontier_mask(m.shared).any()


def test_heuristic_missions_complete_and_explore():
    for nav in ("random", "nearest", "infogain"):
        m = Mission(load_config("flood"), seed=4, reliability="hybrid", nav=nav)
        while not m.done:
            m.step()
        e = m.evaluate()
        assert e["coverage"] > 0.4, nav
        assert m.done_reason in ("max_steps", "all_victims_found", "exploration_complete", "unable_to_continue")
