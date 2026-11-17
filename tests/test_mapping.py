import math

import numpy as np

from config.loader import load_config
from core.mission import Mission
from environment.world import World
from mapping.localization import EKFLocalizer
from mapping.occupancy_grid import OccupancyGrid
from mapping.scan_matching import scan_match
from sensors.suite import SensorSuite


def test_occupancy_update_marks_free_and_occupied():
    cfg = load_config("earthquake")
    w = World(cfg, 0)
    og = OccupancyGrid(w.W, w.H, cfg)
    angles = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    ranges = w.cast_rays(w.x, w.y, w.theta + angles, 8.0)
    for _ in range(3):  # one scan is weak evidence (l_free = -0.4); repeated scans make it certain
        og.update(w.pose, angles, ranges, 8.0)
    assert og.free[int(w.y), int(w.x)]
    assert og.occupied.sum() > 10
    # estimated occupied cells must be real obstacles (ideal sensor, true pose)
    assert w.grid[og.occupied].mean() > 0.95


def test_nan_beams_ignored():
    cfg = load_config("fire")
    og = OccupancyGrid(40, 30, cfg)
    og.update(np.array([5.5, 5.5, 0]), np.zeros(10), np.full(10, np.nan), 8.0)
    assert not og.known.any()


def test_scan_matching_recovers_offset():
    cfg = load_config("earthquake")
    w = World(cfg, 1)
    s = SensorSuite(cfg, 1)
    og = OccupancyGrid(w.W, w.H, cfg)
    rng = np.random.default_rng(0)
    free = np.argwhere(w.reachable)
    for _ in range(40):
        cy, cx = free[rng.integers(len(free))]
        w.x, w.y, w.theta = cx + 0.5, cy + 0.5, rng.uniform(-3, 3)
        o = s.sensors["lidar"].observe(w, w.ground_truth_conditions(), 0, False, 0, {})
        og.update(w.pose, o.data["angles"], o.data["ranges"], 8.0)
    errs_before, errs_after = [], []
    for _ in range(10):
        cy, cx = free[rng.integers(len(free))]
        w.x, w.y, w.theta = cx + 0.5, cy + 0.5, rng.uniform(-3, 3)
        o = s.sensors["lidar"].observe(w, w.ground_truth_conditions(), 0, False, 0, {})
        pert = w.pose + np.array([rng.uniform(-.4, .4), rng.uniform(-.4, .4), rng.uniform(-.06, .06)])
        sm = scan_match(og.prob, int(og.occupied.sum()), pert, o.data["angles"], o.data["ranges"])
        errs_before.append(np.hypot(*(pert[:2] - w.pose[:2])))
        errs_after.append(np.hypot(*(sm["pose"][:2] - w.pose[:2])))
    assert np.median(errs_after) < np.median(errs_before)


def test_ekf_predict_update():
    cfg = load_config("flood")
    ekf = EKFLocalizer([5.0, 5.0, 0.0], cfg)
    r = {"imu": 1.0, "odometry": 1.0}
    ekf.predict({"dist": 1.0, "d_theta": 0.0}, {"yaw_rate": 0.0}, {"rot_cmd": 0.0, "dist_cmd": 1.0}, r)
    assert abs(ekf.pose[0] - 6.0) < 1e-6
    p_before = np.trace(ekf.P)
    ekf.update([6.2, 5.0, 0.0], 1.0, 1.0)
    assert 6.0 < ekf.pose[0] < 6.2 and np.trace(ekf.P) < p_before
    assert not ekf.update([0, 0, 0], 0.0, 1.0)  # unreliable LiDAR -> no update


def test_ekf_uses_command_fallback_when_motion_sensors_fail():
    ekf = EKFLocalizer([5.0, 5.0, 0.0], load_config("fire"))
    ekf.predict({"dist": 0.0, "d_theta": 0.0}, {"yaw_rate": float("nan")}, {"rot_cmd": 0.0, "dist_cmd": 1.0},
                {"imu": 0.0, "odometry": 0.0})
    assert abs(ekf.pose[0] - 6.0) < 1e-6 and np.all(np.isfinite(ekf.P))


def test_localization_not_ground_truth_and_accurate():
    m = Mission(load_config("earthquake"), seed=1, reliability="hybrid", nav="infogain")
    for _ in range(80):
        m.step()
    est, tru = np.array(m.est_traj), np.array(m.true_traj)
    err = np.hypot(est[:, 0] - tru[:, 0], est[:, 1] - tru[:, 1])
    assert err.max() > 0.0          # estimate is not secretly the ground truth
    assert err.mean() < 0.5         # but it tracks it well under nominal conditions
    lay = m.shared.layers()
    assert set(lay) == {"occupancy", "traversability", "hazard", "victim", "unexplored", "uncertainty"}
    for v in lay.values():
        assert np.isfinite(v).all()
