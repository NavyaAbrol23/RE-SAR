import math

import numpy as np
import pytest

from config.loader import load_config
from environment.world import World
from sensors.base_sensor import SENSOR_NAMES
from sensors.degradation import LEVELS
from sensors.suite import SensorSuite


def _setup(disaster="earthquake", seed=1):
    cfg = load_config(disaster)
    return World(cfg, seed), SensorSuite(cfg, seed)


def test_all_six_sensors_return_standard_observations():
    w, s = _setup()
    obs = s.observe(w, 0, {"actual_rot": 0.1, "actual_dist": 1.0, "rot_cmd": 0.1, "dist_cmd": 1.0})
    assert set(obs) == set(SENSOR_NAMES)
    for name, o in obs.items():
        assert o.name == name and o.delivered and isinstance(o.data, dict)
    assert obs["rgb"].data["image"].dtype == np.uint8 and obs["rgb"].data["image"].ndim == 3
    assert obs["thermal"].data["temp"].dtype == np.float32
    assert len(obs["lidar"].data["ranges"]) == 72
    assert math.isfinite(obs["imu"].data["yaw"]) and math.isfinite(obs["odometry"].data["dist"])
    assert "detections" in obs["audio"].data


@pytest.mark.parametrize("sensor", SENSOR_NAMES)
def test_complete_failure_modes(sensor):
    w, s = _setup()
    s.degradation.set_level(sensor, "failed")
    ctx = {"actual_rot": 0.3, "actual_dist": 1.0, "rot_cmd": 0.3, "dist_cmd": 1.0}
    for t in range(3):
        o = s.observe(w, t, ctx)[sensor]
    d = o.data
    if sensor == "rgb":
        assert d["image"].max() == 0
    elif sensor == "thermal":
        assert np.all(d["temp"][d["fov_mask"]] >= d["t_sat"] - 1e-3)
    elif sensor == "lidar":
        assert np.isnan(d["ranges"]).all()
    elif sensor == "imu":
        assert d["frozen"] and d["yaw_rate"] == 0.0
    elif sensor == "odometry":
        assert d["dist"] == 0.0 and d["d_theta"] == 0.0
    elif sensor == "audio":
        assert d["noise_floor"] > 1.0
    assert s.true_reliability[sensor] == 0.0


def test_degradation_levels_monotone_truth():
    w, s = _setup("fire")
    vals = []
    for lvl in LEVELS:
        s.degradation.set_level("lidar", lvl)
        s.observe(w, 0, {})
        vals.append(s.true_reliability["lidar"])
    assert vals == sorted(vals, reverse=True) and vals[-1] == 0.0


def test_simultaneous_degradation():
    w, s = _setup("flood")
    s.degradation.set_levels({"lidar": "severe", "odometry": "severe", "audio": "failed"})
    s.observe(w, 0, {})
    tr = s.true_reliability
    assert tr["audio"] == 0.0 and tr["lidar"] < 0.5 and tr["odometry"] < 0.5


def test_lidar_noise_and_dropouts_increase_with_degradation():
    w, s = _setup()
    nan_frac = {}
    for lvl in ("normal", "severe"):
        s.degradation.set_level("lidar", lvl)
        fr = [np.isnan(s.observe(w, t, {})["lidar"].data["ranges"]).mean() for t in range(20)]
        nan_frac[lvl] = np.mean(fr)
    assert nan_frac["severe"] > nan_frac["normal"] + 0.2


def test_rgb_degradation_reduces_contrast():
    w, s = _setup("fire")
    from reliability.observation_quality import rgb_quality
    q = {}
    for lvl in ("normal", "severe"):
        s.degradation.set_level("rgb", lvl)
        img = s.observe(w, 0, {})["rgb"].data
        q[lvl] = rgb_quality(img["image"], img["fov_mask"])["q"]
    assert q["severe"] < q["normal"]


def test_thermal_shows_victim_heat():
    w, s = _setup("earthquake", 3)
    v = w.victims[0]
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        x, y = v.x - 2.5 * math.cos(ang), v.y - 2.5 * math.sin(ang)
        if not w.blocked(x, y) and w.visible_mask(x, y, ang, 2.0, 6)[int(v.y), int(v.x)]:
            w.x, w.y, w.theta = x, y, ang
            T = s.observe(w, 0, {})["thermal"].data["temp"]
            assert T.max() > 30.0
            return
    pytest.skip("no clear view of victim on this map")
