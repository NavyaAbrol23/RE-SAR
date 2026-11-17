import itertools
import math

import numpy as np
import pytest

from config.loader import load_config
from core.mission import Mission
from reliability.cross_modal import CrossModalAgreement
from reliability.environment_state import EnvironmentStateEstimator
from reliability.hybrid_fusion import STRATEGIES, HybridReliability, normalise
from reliability.observation_quality import audio_quality, lidar_quality, rgb_quality, thermal_quality
from sensors.base_sensor import LOCALIZATION_SENSORS, PERCEPTION_SENSORS, SENSOR_NAMES

ONES = {s: 1.0 for s in SENSOR_NAMES}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_scores_in_unit_interval(strategy):
    cfg = load_config("fire")
    hr = HybridReliability(cfg, strategy)
    rng = np.random.default_rng(0)
    for _ in range(50):
        comp = [{s: float(rng.uniform(-0.5, 1.5)) for s in SENSOR_NAMES} for _ in range(4)]
        rep = hr.update(*comp)
        for s in SENSOR_NAMES:
            assert 0.0 <= rep.r[s] <= 1.0 and math.isfinite(rep.r[s])
        for w in (rep.weights_loc, rep.weights_perc):
            assert all(0 <= v <= 1 for v in w.values())


def test_failed_sensor_gets_zero_weight_and_degraded_keeps_some():
    cfg = load_config("earthquake")
    hr = HybridReliability(cfg, "hybrid")
    health = dict(ONES, lidar=0.0)
    quality = dict(ONES, imu=0.3)
    rep = hr.update(ONES, quality, health, ONES)
    assert rep.r["lidar"] == 0.0 and rep.weights_loc["lidar"] == 0.0
    assert 0 < rep.weights_loc["imu"] < rep.weights_loc["odometry"]
    assert abs(sum(rep.weights_loc.values()) - 1) < 1e-9


def test_all_sensors_failed_safe_fallback():
    cfg = load_config("flood")
    hr = HybridReliability(cfg, "hybrid")
    zeros = {s: 0.0 for s in SENSOR_NAMES}
    rep = hr.update(ONES, ONES, zeros, ONES)
    assert rep.loc_all_failed and rep.perc_all_failed
    assert all(v == 0.0 for v in rep.weights_loc.values())
    w, failed = normalise({s: float("nan") for s in SENSOR_NAMES}, LOCALIZATION_SENSORS)
    assert failed and all(v == 0 for v in w.values())


def test_fixed_strategy_ignores_evidence():
    hr = HybridReliability(load_config("fire"), "fixed")
    zeros = {s: 0.0 for s in SENSOR_NAMES}
    rep = hr.update(zeros, zeros, zeros, zeros)
    assert all(v == 1.0 for v in rep.r.values())


def test_environment_prior_reacts_to_conditions():
    est = EnvironmentStateEstimator(load_config("fire", {"world": {"severity": 0.9}}))
    calm = est.reliability(est.estimate_conditions(0.0, 0.0, 0.0))
    hot = est.reliability(est.estimate_conditions(0.9, 0.0, 0.0))
    assert hot["thermal"] < calm["thermal"] and hot["rgb"] < calm["rgb"]


def test_quality_metrics_detect_bad_observations():
    fov = np.ones((50, 50), bool)
    good = (np.random.default_rng(0).random((50, 50, 3)) * 200).astype(np.uint8)
    assert rgb_quality(np.zeros_like(good), fov)["q"] == 0.0
    assert rgb_quality(good, fov)["q"] > 0.3
    T = np.full((50, 50), 120.0, np.float32)
    assert thermal_quality(T, 120.0, fov)["q"] == 0.0
    assert lidar_quality(np.full(72, np.nan), 8.0)["q"] == 0.0
    assert lidar_quality(np.full(72, 3.0), 8.0)["q"] == 1.0
    assert audio_quality(2.0)["q"] == 0.0 and audio_quality(0.03)["q"] > 0.9


def test_cross_modal_only_compares_related_sensors():
    cm = CrossModalAgreement(gate=2.0)
    cands = {"rgb": [{"x": 5, "y": 5}], "thermal": [{"x": 5.5, "y": 5.2}], "audio": []}
    for _ in range(10):
        cm.update_perception(cands, [], (0, 0, 0), {"rgb": True, "thermal": True, "audio": True})
    assert cm.a["rgb"] > 0.9 and cm.a["thermal"] > 0.9
    cm2 = CrossModalAgreement(gate=2.0)
    for _ in range(10):  # partner unusable -> no forced comparison, agreement stays neutral
        cm2.update_perception(cands, [], (0, 0, 0), {"rgb": True, "thermal": False, "audio": True})
    assert abs(cm2.a["rgb"] - 0.5) < 0.1


@pytest.mark.parametrize("failed", list(SENSOR_NAMES) + [("rgb", "lidar", "thermal")])
def test_mission_detects_failures_without_nan(failed):
    failed = failed if isinstance(failed, tuple) else (failed,)
    m = Mission(load_config("earthquake"), seed=2, reliability="hybrid", nav="infogain",
                degradation={s: "failed" for s in failed})
    for _ in range(40):
        m.step()
        assert np.all(np.isfinite(m.ekf.pose))
        for s in SENSOR_NAMES:
            assert 0.0 <= m.rel.r[s] <= 1.0
    for s in failed:
        assert m.rel.r[s] == 0.0, s


def test_all_six_failed_does_not_crash():
    m = Mission(load_config("fire"), seed=0, reliability="hybrid", nav="nearest",
                degradation={s: "failed" for s in SENSOR_NAMES})
    for _ in range(30):
        m.step()
    assert np.all(np.isfinite(m.ekf.pose))
    rep = m.last["rep"]
    assert rep.perc_all_failed
    assert rep.r["lidar"] == 0.0 and rep.weights_loc["lidar"] == 0.0
    # note: a frozen IMU is only detectable while the robot is commanded to turn; with every
    # exteroceptive sensor dead the robot never plans a motion, so the IMU may still look healthy.


def test_learned_strategy_if_model_present():
    from reliability.learned import model_available
    if not model_available():
        pytest.skip("B6 model not trained (python -m reliability.learned)")
    hr = HybridReliability(load_config("fire"), "learned")
    rep = hr.update(ONES, ONES, dict(ONES, rgb=0.0), ONES)
    assert rep.r["rgb"] == 0.0 and all(0 <= v <= 1 for v in rep.r.values())
