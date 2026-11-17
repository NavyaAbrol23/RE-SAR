"""(E) Hybrid reliability and normalised, group-wise fusion weights.

For sensor s with environment prior e_s, observation quality q_s, health h_s and cross-modal
agreement a_s (all in [0,1]):

  fixed               r_s = 1
  env                 r_s = e_s
  quality             r_s = q_s
  env_quality         r_s = (w_e e_s + w_q q_s) / (w_e + w_q)
  env_quality_health  r_s = h_s * (w_e e_s + w_q q_s) / (w_e + w_q)
  hybrid              r_s = h_s * (w_e e_s + w_q q_s + w_c a_s) / (w_e + w_q + w_c)
  learned (B6)        r_s = h_s * MLP(e_s, q_s, h_s, a_s, onehot(s))   (see reliability/learned.py)

Temporal smoothing: r_t = (1 - k) r_{t-1} + k r_raw, except that h_s = 0 forces r_s = 0 at once.
Fusion weights are normalised separately inside the localisation group {lidar, imu, odometry}
and the perception group {rgb, thermal, audio}:  w_s = r_s / sum_group r.
If sum_group r < eps the group is flagged as `all_failed` and callers fall back safely.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sensors.base_sensor import LOCALIZATION_SENSORS, PERCEPTION_SENSORS, SENSOR_NAMES

STRATEGIES = ("fixed", "env", "quality", "env_quality", "env_quality_health", "hybrid")
EXTRA_STRATEGIES = ("learned",)  # B6: needs rl/models/reliability_mlp.npz (python -m reliability.learned)
EPS = 1e-6


@dataclass
class ReliabilityReport:
    strategy: str
    r: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)
    agreement: dict = field(default_factory=dict)
    weights_loc: dict = field(default_factory=dict)
    weights_perc: dict = field(default_factory=dict)
    loc_all_failed: bool = False
    perc_all_failed: bool = False


def normalise(r: dict, group) -> tuple[dict, bool]:
    tot = float(sum(max(0.0, r[s]) for s in group))
    if not np.isfinite(tot) or tot < EPS:
        return {s: 0.0 for s in group}, True
    return {s: max(0.0, r[s]) / tot for s in group}, False


class HybridReliability:
    def __init__(self, cfg: dict, strategy: str | None = None):
        rc = cfg["reliability"]
        self.strategy = strategy or rc["strategy"]
        if self.strategy not in STRATEGIES + EXTRA_STRATEGIES:
            raise ValueError(f"unknown reliability strategy {self.strategy}")
        self.learned = None
        if self.strategy == "learned":
            from reliability.learned import LearnedReliability, model_available
            if not model_available():
                raise FileNotFoundError("learned reliability model missing: run python -m reliability.learned")
            self.learned = LearnedReliability()
        self.w_e, self.w_q, self.w_c = rc["w_env"], rc["w_quality"], rc["w_cross"]
        self.k = rc["smoothing"]
        self.r = {s: 1.0 for s in SENSOR_NAMES}

    def reset(self):
        self.r = {s: 1.0 for s in SENSOR_NAMES}

    def raw(self, s, e, q, h, a) -> float:
        st = self.strategy
        if st == "fixed":
            return 1.0
        if st == "env":
            return e
        if st == "quality":
            return q
        eq = (self.w_e * e + self.w_q * q) / (self.w_e + self.w_q)
        if st == "env_quality":
            return eq
        if st == "env_quality_health":
            return h * eq
        return h * (self.w_e * e + self.w_q * q + self.w_c * a) / (self.w_e + self.w_q + self.w_c)

    def update(self, env: dict, quality: dict, health: dict, agreement: dict) -> ReliabilityReport:
        if self.learned is not None:
            from reliability.learned import features
            X = np.array([features(env[s], quality[s], health[s], agreement[s], s) for s in SENSOR_NAMES], np.float32)
            learned = dict(zip(SENSOR_NAMES, self.learned.predict(X)))
        for s in SENSOR_NAMES:
            if self.learned is not None:
                v = health[s] * float(learned[s]) if health[s] > 0.05 else 0.0
            else:
                v = self.raw(s, env[s], quality[s], health[s], agreement[s])
            v = float(np.clip(v if np.isfinite(v) else 0.0, 0.0, 1.0))
            uses_health = self.strategy in ("env_quality_health", "hybrid", "learned")
            if uses_health and health[s] <= 0.05:
                self.r[s] = 0.0
            elif self.strategy == "fixed":
                self.r[s] = 1.0
            else:
                self.r[s] = float(np.clip((1 - self.k) * self.r[s] + self.k * v, 0.0, 1.0))
        wl, lf = normalise(self.r, LOCALIZATION_SENSORS)
        wp, pf = normalise(self.r, PERCEPTION_SENSORS)
        return ReliabilityReport(self.strategy, dict(self.r), dict(env), dict(quality), dict(health),
                                 dict(agreement), wl, wp, lf, pf)
