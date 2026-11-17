"""GUI-facing simulation controller: owns a Mission and exposes the actions behind every button."""
from __future__ import annotations

import time

import numpy as np

from config.loader import load_config
from core.mission import Mission
from reliability.hybrid_fusion import STRATEGIES, HybridReliability
from rl.policy import try_load
from rl.reward import compute_reward
from sensors.base_sensor import SENSOR_NAMES

NAV_MODES = ("random", "nearest", "infogain", "ppo")
DIFFICULTIES = ("easy", "medium", "hard")


class Settings:
    def __init__(self):
        self.disaster = "earthquake"
        self.difficulty = "medium"
        self.severity = 0.5
        self.victims = 5
        self.seed = 42
        self.fusion = "hybrid"
        self.nav = "infogain"


class SimulationController:
    def __init__(self, settings: Settings):
        self.s = settings
        self.paused = False
        self.speed = 2          # low-level steps per frame
        self.message = ""
        self._chooser_cache = {}
        self.build()

    # ---------------------------------------------------------------- build
    def build(self, keep_degradation: bool = False):
        old = dict(self.mission.degradation_levels) if keep_degradation and hasattr(self, "mission") else {}
        cfg = load_config(self.s.disaster, {"seed": self.s.seed, "world": {
            "difficulty": self.s.difficulty, "severity": float(self.s.severity),
            "num_victims": int(self.s.victims)}})
        self.cfg = cfg
        nav, chooser = self._nav_setup(self.s.nav)
        self.mission = Mission(cfg, seed=self.s.seed, reliability=self.s.fusion, nav=nav,
                               degradation=old, chooser=chooser)
        self.cum_reward = 0.0
        self.last_reward = 0.0
        self.metrics = self.mission.evaluate()
        self._last_eval = time.time()
        self.fps_steps = []

    def _nav_setup(self, mode):
        if mode != "ppo":
            return mode, None
        rel = "hybrid" if self.s.fusion != "fixed" else "fixed"
        if rel not in self._chooser_cache:
            self._chooser_cache[rel] = try_load(rel) or try_load("hybrid" if rel == "fixed" else "fixed")
        ch = self._chooser_cache[rel]
        if ch is None:
            self.message = "No trained PPO model found (rl/models/ppo_*.zip) - using info-gain heuristic"
            return "infogain", None
        self.message = f"PPO policy loaded: {ch.path.split('/')[-1].split(chr(92))[-1]}"
        return "ppo", ch

    # -------------------------------------------------------------- actions
    def toggle_pause(self):
        self.paused = not self.paused

    def step_once(self):
        self._advance(1)

    def reset(self):
        self.build(keep_degradation=True)

    def new_map(self):
        self.s.seed += 1
        self.build(keep_degradation=True)

    def cycle_nav(self):
        i = NAV_MODES.index(self.s.nav) if self.s.nav in NAV_MODES else 0
        self.s.nav = NAV_MODES[(i + 1) % len(NAV_MODES)]
        nav, chooser = self._nav_setup(self.s.nav)
        self.mission.nav, self.mission.chooser = nav, chooser
        self.mission.target, self.mission.path = None, []

    def cycle_fusion(self):
        i = STRATEGIES.index(self.s.fusion)
        self.s.fusion = STRATEGIES[(i + 1) % len(STRATEGIES)]
        self.mission.rel_strategy = self.s.fusion
        self.mission.rel = HybridReliability(self.cfg, self.s.fusion)
        if self.s.nav == "ppo":
            nav, chooser = self._nav_setup("ppo")
            self.mission.nav, self.mission.chooser = nav, chooser

    def cycle_sensor(self, name):
        lvl = self.mission.suite.degradation.cycle(name)
        self.mission.degradation_levels[name] = lvl
        return lvl

    def faster(self):
        self.speed = min(20, self.speed + 1)

    def slower(self):
        self.speed = max(1, self.speed - 1)

    # --------------------------------------------------------------- update
    def _advance(self, n):
        m = self.mission
        for _ in range(n):
            if m.done:
                break
            cov0 = m.shared.coverage_estimate()
            last = m.step()
            cx, cy = m.est_cell()
            acc = {"info_gain": last["info_gain"], "new_victims": last["new_victims"],
                   "coverage_delta": m.shared.coverage_estimate() - cov0, "travel": last["cmd"]["dist_cmd"],
                   "hazard": float(m.shared.hazard[cy, cx]), "collisions": int(last["collided"])}
            self.last_reward = compute_reward(acc, self.cfg)
            self.cum_reward += self.last_reward
            self.fps_steps.append(time.time())
        if time.time() - self._last_eval > 0.3 or m.done:
            self.metrics = m.evaluate()
            self._last_eval = time.time()

    def update(self):
        if not self.paused:
            self._advance(self.speed)

    @property
    def steps_per_second(self):
        now = time.time()
        self.fps_steps = [t for t in self.fps_steps if now - t < 2.0]
        return len(self.fps_steps) / 2.0

    def sensor_rows(self):
        m = self.mission
        rep = m.last["rep"]
        tr = m.suite.true_reliability
        rows = []
        for s in SENSOR_NAMES:
            obs = m.last["obs"][s] if m.last["obs"] else None
            rows.append({"name": s, "r": rep.r[s], "env": rep.env[s], "q": rep.quality[s],
                         "h": rep.health[s], "a": rep.agreement[s], "true": tr[s],
                         "level": m.suite.degradation.levels[s], "status": self._status(s, obs)})
        return rows

    def _status(self, s, obs):
        if obs is None:
            return "-"
        d = obs.data
        if s == "lidar":
            r = d["ranges"]
            return f"{int(np.isfinite(r).sum())}/{len(r)} returns, {int(np.isnan(r).sum())} missing"
        if s == "imu":
            return f"yaw {d['yaw']:+.2f} rate {d['yaw_rate']:+.2f}" + (" FROZEN" if d.get("frozen") else "")
        if s == "odometry":
            return f"d {d['dist']:.2f}  dth {d['d_theta']:+.2f}"
        if s == "audio":
            return f"{len(d['detections'])} det, floor {d['noise_floor']:.2f}"
        if s == "rgb":
            n = len(self.mission.last["cands"].get("rgb", []))
            return f"mean {d['image'][d['fov_mask']].mean():.0f}, {n} cand"
        if s == "thermal":
            n = len(self.mission.last["cands"].get("thermal", []))
            return f"sat {d['t_sat']:.0f}C, {n} cand"
        return ""
