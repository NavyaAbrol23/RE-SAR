"""SAREnvironment: Gymnasium wrapper around the RE-SAR Mission.

The agent selects EXPLORATION TARGETS, not motor commands:
  action a in {0..K-1}: go to frontier slot a (slots ordered by path cost, nearest first)
  action K           : STOP / end the mission
An action pointing at an empty slot is invalid: it is remapped to slot 0 (nearest frontier)
and penalised. A* and the safety controller then execute up to `macro_steps` low-level moves.

Each reset draws a new procedural map (disaster, severity, seed) and, with probability
`p_degrade`, random sensor degradations, so the policy is trained under domain randomisation.
Training seeds and evaluation seeds come from disjoint ranges (unseen maps at evaluation).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.loader import DISASTERS, deep_merge, load_config
from core.mission import Mission
from rl.observation import K_FRONTIERS, OBS_DIM, build_observation
from rl.reward import compute_reward
from sensors.base_sensor import SENSOR_NAMES

TRAIN_SEED_BASE = 0
EVAL_SEED_BASE = 10_000_000


class SAREnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reliability: str = "hybrid", disasters=DISASTERS, severity_range=(0.2, 0.8),
                 p_degrade: float = 0.6, seed_base: int = TRAIN_SEED_BASE, fixed_degradation=None,
                 overrides: dict | None = None, seed: int | None = None):
        super().__init__()
        self.reliability = reliability
        self.disasters = tuple(disasters)
        self.severity_range = severity_range
        self.p_degrade = p_degrade
        self.seed_base = seed_base
        self.fixed_degradation = fixed_degradation
        self.overrides = overrides or {}
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(K_FRONTIERS + 1)
        self._init_seed = seed
        self.mission: Mission | None = None
        self.episode_seed = None

    # ------------------------------------------------------------------
    def _sample_degradation(self):
        if self.fixed_degradation is not None:
            return dict(self.fixed_degradation)
        if self._rng.random() > self.p_degrade:
            return {}
        n = int(self._rng.integers(1, 3))
        sens = self._rng.choice(SENSOR_NAMES, size=n, replace=False)
        return {str(s): str(self._rng.choice(["mild", "severe", "failed"], p=[0.35, 0.35, 0.3])) for s in sens}

    def reset(self, *, seed=None, options=None):
        if seed is None and self._init_seed is not None:
            seed, self._init_seed = self._init_seed, None
        super().reset(seed=seed)
        self._rng = self.np_random
        options = options or {}
        disaster = options.get("disaster") or str(self._rng.choice(self.disasters))
        severity = options.get("severity", float(self._rng.uniform(*self.severity_range)))
        self.episode_seed = options.get("map_seed", self.seed_base + int(self._rng.integers(0, 1_000_000)))
        cfg = load_config(disaster, deep_merge(self.overrides, {"world": {"severity": severity}}))
        deg = options.get("degradation", self._sample_degradation())
        self.mission = Mission(cfg, seed=self.episode_seed, reliability=self.reliability, nav="external",
                               degradation=deg)
        self.mission.compute_frontiers()
        self.cfg = cfg
        return build_observation(self.mission), self._info()

    def _info(self):
        m = self.mission
        return {"disaster": m.cfg["disaster"], "seed": self.episode_seed, "t": m.t,
                "confirmed": len(m.tracker.confirmed), "n_frontiers": len(m.frontiers),
                "degradation": dict(m.degradation_levels)}

    def step(self, action):
        m = self.mission
        action = int(action)
        acc = {"info_gain": 0.0, "new_victims": 0, "coverage_delta": 0.0, "travel": 0.0,
               "hazard": 0.0, "collisions": 0, "invalid": 0, "stop_penalty": 0.0}
        cov0 = m.shared.coverage_estimate()
        terminated = truncated = False
        if action == K_FRONTIERS:
            terminated = True
            if m.frontiers:
                acc["stop_penalty"] = 2.0 + 5.0 * (1.0 - cov0)
            m.done, m.done_reason = True, "agent_stop"
        else:
            if action >= len(m.frontiers):
                acc["invalid"] = 1
                action = 0
            ok = m.plan(action) if m.frontiers else False
            if not ok:
                # still advance time so a degenerate policy cannot stall the clock
                m.step()
                acc["travel"] += 0.5
            else:
                for _ in range(self.cfg["rl"]["macro_steps"]):
                    last = m.step()
                    acc["info_gain"] += last["info_gain"]
                    acc["new_victims"] += last["new_victims"]
                    acc["travel"] += last["cmd"]["dist_cmd"]
                    cx, cy = m.est_cell()
                    acc["hazard"] += float(m.shared.hazard[cy, cx])
                    acc["collisions"] += int(last["collided"])
                    if m.done or m.target is None or last["new_victims"]:
                        break
            acc["coverage_delta"] = m.shared.coverage_estimate() - cov0
            fr = m.compute_frontiers()
            if m.done:
                truncated = m.done_reason == "max_steps"
                terminated = not truncated
            elif not fr:
                terminated = True
                m.done, m.done_reason = True, "exploration_complete"
        reward = compute_reward(acc, self.cfg)
        info = self._info()
        info["reward_terms"] = acc
        return build_observation(m), reward, terminated, truncated, info
