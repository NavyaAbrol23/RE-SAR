"""The RE-SAR perception-action loop, shared by the GUI, the Gymnasium env and the experiments.

Per timestep:
  1. choose/keep an exploration target (heuristic, PPO chooser or external RL action)
  2. A* path on the shared costmap -> next waypoint -> safety check -> rotate+drive command
  3. ground-truth world executes the command (slip, collisions, hazards)
  4. six sensors observe (with degradation)
  5. health, observation quality, environment prior, cross-modal agreement -> reliability
  6. EKF predict (reliability-weighted IMU/odometry) -> scan matching -> EKF update
  7. occupancy + hazard mapping at the ESTIMATED pose
  8. victim candidates (RGB / thermal / audio) -> association -> reliability-weighted tracks
  9. shared map refresh
Ground truth (`self.world`) is only read for sensor generation and the `evaluate()` metrics.
"""
from __future__ import annotations

import math
import time

import numpy as np

from environment.world import World, wrap
from mapping.localization import EKFLocalizer
from mapping.scan_matching import scan_match
from mapping.shared_map import SharedMap
from navigation.astar import astar, dijkstra
from navigation.costmap import build_costmap
from navigation.frontier import find_frontiers, frontier_mask, select_infogain, select_nearest, select_random
from navigation.safety import SafetyController
from perception.audio_detection import detect_audio
from perception.data_association import to_world
from perception.rgb_detection import detect_rgb
from perception.thermal_detection import detect_thermal
from perception.victim_tracking import VictimTracker
from reliability.cross_modal import CrossModalAgreement
from reliability.environment_state import EnvironmentStateEstimator
from reliability.hybrid_fusion import HybridReliability
from reliability.observation_quality import (ResidualTracker, audio_quality, lidar_quality,
                                             rgb_quality, thermal_quality)
from reliability.sensor_health import SensorHealthMonitor
from sensors.base_sensor import SENSOR_NAMES
from sensors.suite import SensorSuite

NAV_STRATEGIES = ("random", "nearest", "infogain", "ppo", "external")


class Mission:
    def __init__(self, cfg: dict, seed: int | None = None, reliability: str | None = None,
                 nav: str | None = None, degradation: dict | None = None,
                 modalities=("rgb", "thermal", "audio"), chooser=None):
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 0) if seed is None else seed)
        self.rel_strategy = reliability or cfg["reliability"]["strategy"]
        self.nav = nav or cfg["navigation"]["strategy"]
        if self.nav not in NAV_STRATEGIES:
            raise ValueError(f"unknown navigation strategy {self.nav}")
        self.modalities = tuple(modalities)
        self.chooser = chooser
        self.degradation_levels = dict(degradation or {})
        self.world = World(cfg, self.seed)
        self.suite = SensorSuite(cfg, self.seed)
        self.suite.degradation.set_levels(self.degradation_levels)
        W, H = self.world.W, self.world.H
        self.shared = SharedMap(W, H, cfg)
        self.ekf = EKFLocalizer(self.world.pose, cfg)
        self.health = SensorHealthMonitor()
        self.residuals = ResidualTracker()
        self.cross = CrossModalAgreement(cfg["perception"]["gate"])
        self.env_est = EnvironmentStateEstimator(cfg)
        self.rel = HybridReliability(cfg, self.rel_strategy)
        self.tracker = VictimTracker(cfg)
        self.safety = SafetyController(cfg)
        self.reset()

    # ================================================================== setup
    def reset(self):
        self.rng = np.random.default_rng(self.seed + 101)
        self.world.reset_robot()
        self.suite.reset()
        self.shared.reset()
        self.ekf.reset(self.world.pose)
        self.health.reset()
        self.residuals = ResidualTracker()
        self.cross.reset()
        self.rel.reset()
        self.tracker.reset()
        self.safety.vetoes = 0
        self.t = 0
        self.done, self.done_reason = False, ""
        self.target, self.path, self.frontiers = None, [], []
        self.target_index, self.last_action_source = None, "-"
        self.blacklist: dict = {}
        self.plan_failures = 0
        self.steps_since_plan = 0
        self.est_traj, self.true_traj = [], []
        self.rel_log_est, self.rel_log_true = [], []
        self.step_times = []
        self.true_path_len = 0.0
        self.total_new_victims = 0
        self.last = {"obs": None, "rep": None, "cands": {}, "cmd": {"rot_cmd": 0.0, "dist_cmd": 0.0},
                     "info_gain": 0.0, "new_victims": 0, "exposure": 0.0, "collided": False,
                     "safety": "ok"}
        cmd = {"rot_cmd": 0.0, "dist_cmd": 0.0}
        for _ in range(2):  # stationary start-up scans so the first frontiers are well established
            obs = self.suite.observe(self.world, 0, {"actual_rot": 0.0, "actual_dist": 0.0, **cmd})
            self._process(obs, cmd)

    def set_degradation(self, sensor: str, level: str):
        self.suite.degradation.set_level(sensor, level)
        self.degradation_levels[sensor] = level

    # =============================================================== planning
    def est_cell(self):
        x, y, _ = self.ekf.pose
        return (int(np.clip(math.floor(x), 0, self.world.W - 1)), int(np.clip(math.floor(y), 0, self.world.H - 1)))

    def compute_frontiers(self):
        cost = build_costmap(self.shared, self.cfg)
        sc = self.est_cell()
        cost[sc[1], sc[0]] = 1.0  # the robot's own cell is always traversable
        dist = dijkstra(cost, sc)
        bl = {c for c, n in self.blacklist.items() if n >= 2}
        self.frontiers = find_frontiers(self.shared, dist, self.cfg, bl)[: self.cfg["navigation"]["max_frontiers"]]
        self._cost = cost
        return self.frontiers

    def plan(self, index: int | None = None) -> bool:
        """Select a frontier (by strategy or given index) and compute an A* path to it."""
        fr = self.compute_frontiers()
        self.steps_since_plan = 0
        if not fr:
            self.target, self.path = None, []
            self.plan_failures += 1
            return False
        if index is None:
            if self.nav == "random":
                index, src = select_random(fr, self.rng), "random"
            elif self.nav == "nearest":
                index, src = select_nearest(fr), "nearest"
            elif self.nav == "ppo" and self.chooser is not None:
                index, src = int(self.chooser(self)), "ppo"
            else:
                index, src = select_infogain(fr, self.cfg), "infogain"
        else:
            src = "rl"
        index = int(np.clip(index, 0, len(fr) - 1))
        f = fr[index]
        path = astar(self._cost, self.est_cell(), (f.x, f.y))
        if path is None:
            self.blacklist[(f.x, f.y)] = self.blacklist.get((f.x, f.y), 0) + 2
            self.target, self.path = None, []
            self.plan_failures += 1
            return False
        self.target, self.path, self.target_index, self.last_action_source = f, path, index, src
        self.plan_failures = 0
        return True

    def needs_decision(self) -> bool:
        return (not self.done) and (self.target is None)

    # ================================================================== step
    def step(self) -> dict:
        if self.done:
            return self.last
        t0 = time.perf_counter()
        n = self.cfg["navigation"]
        if self.nav != "external" and (self.target is None or self.steps_since_plan >= n["replan_every"]):
            self.plan()
        cmd = {"rot_cmd": 0.0, "dist_cmd": 0.0}
        safety_msg = "idle"
        if self.target is not None and self.path:
            nxt = self.path[0]
            r_l = self.rel.r["lidar"]
            ok, safety_msg = self.safety.check(self.shared, self.ekf.pose, nxt,
                                               self.last["obs"]["lidar"].data if self.last["obs"] else None, r_l)
            if not ok:
                new = astar(self._cost_now(), self.est_cell(), (self.target.x, self.target.y))
                if new is None or (new and new[0] == nxt):
                    key = (self.target.x, self.target.y)
                    self.blacklist[key] = self.blacklist.get(key, 0) + 1
                    self.target, self.path = None, []
                else:
                    self.path = new
            if self.target is not None and self.path:
                x, y, th = self.ekf.pose
                wx, wy = self.path[0][0] + 0.5, self.path[0][1] + 0.5
                heading = math.atan2(wy - y, wx - x)
                rot = wrap(heading - th)
                dist = min(math.hypot(wx - x, wy - y), 1.0)
                if abs(rot) > math.radians(60):  # turn in place first on sharp turns
                    dist = 0.0
                cmd = {"rot_cmd": rot, "dist_cmd": dist}
        prev_true = (self.world.x, self.world.y)
        out = self.world.apply_motion(cmd["rot_cmd"], cmd["dist_cmd"], self.rng)
        self.true_path_len += math.hypot(self.world.x - prev_true[0], self.world.y - prev_true[1])
        self.t += 1
        self.steps_since_plan += 1
        obs = self.suite.observe(self.world, self.t, {**out, **cmd})
        self._process(obs, cmd)
        self.last.update({"exposure": out["exposure"], "collided": out["collided"], "safety": safety_msg})
        # waypoint / target bookkeeping on the ESTIMATED pose
        if self.target is not None and self.path:
            x, y, _ = self.ekf.pose
            if math.hypot(self.path[0][0] + 0.5 - x, self.path[0][1] + 0.5 - y) < 0.45:
                self.path.pop(0)
            if not self.path:
                self.target = None
        elif self.target is not None:
            self.target = None
        self._check_done()
        self.step_times.append(time.perf_counter() - t0)
        return self.last

    def _cost_now(self):
        cost = build_costmap(self.shared, self.cfg)
        sc = self.est_cell()
        cost[sc[1], sc[0]] = 1.0
        self._cost = cost
        return cost

    def _check_done(self):
        n_exp = int(self.cfg["world"]["num_victims"])
        if self.t >= self.cfg["robot"]["max_steps"]:
            self.done, self.done_reason = True, "max_steps"
        elif len(self.tracker.confirmed) >= n_exp and self.shared.coverage_estimate() >= 0.8:
            self.done, self.done_reason = True, "all_victims_found"
        elif self.plan_failures >= 4:
            self.done = True
            self.done_reason = "unable_to_continue" if frontier_mask(self.shared).sum() >= 3 else "exploration_complete"

    # ============================================================ perception
    def _process(self, obs: dict, cmd: dict):
        st = self.rel_strategy
        imu, odo, lid = obs["imu"].data, obs["odometry"].data, obs["lidar"].data
        h = self.health.assess(obs, cmd)
        rgbq = rgb_quality(obs["rgb"].data["image"], obs["rgb"].data["fov_mask"])
        thq = thermal_quality(obs["thermal"].data["temp"], obs["thermal"].data["t_sat"],
                              obs["thermal"].data["fov_mask"], self.cfg["sensors"]["thermal"]["t_sat"])
        q = {"rgb": rgbq["q"], "thermal": thq["q"],
             "lidar": lidar_quality(lid["ranges"], lid["max_range"])["q"],
             "imu": self.residuals.update_imu(imu["yaw_rate"], cmd["rot_cmd"])["q"],
             "odometry": self.residuals.update_odo(odo["dist"], odo["d_theta"], cmd["dist_cmd"], cmd["rot_cmd"])["q"],
             "audio": audio_quality(obs["audio"].data["noise_floor"])["q"]}
        x, y, _ = self.ekf.pose
        cond = self.env_est.estimate_conditions(*self.shared.terrain.local(x, y))
        e = self.env_est.reliability(cond)
        rep = self.rel.update(e, q, h, dict(self.cross.a))
        r = rep.r
        prev_pose = self.ekf.pose
        self.ekf.predict(odo, imu, cmd, r)
        sm = None
        if r["lidar"] > 0.02:
            n_occ = int(self.shared.occ.occupied.sum())
            sm = scan_match(self.shared.occ.prob, n_occ, self.ekf.pose, lid["angles"], lid["ranges"],
                            lid["max_range"], sigma_xy=self.ekf.sigma_xy, sigma_th=math.sqrt(self.ekf.P[2, 2]))
            if sm["ok"]:
                sm["dist"] = float(math.hypot(sm["pose"][0] - prev_pose[0], sm["pose"][1] - prev_pose[1]))
                sm["dtheta"] = float(wrap(sm["pose"][2] - prev_pose[2]))
                mq = 1.0 if st == "fixed" else float(np.clip((sm["score"] - 0.3) / 0.5, 0, 1))
                self.ekf.update(sm["pose"], r["lidar"], mq)
        self.cross.update_localization(odo, imu, sm, h)
        pose = self.ekf.pose
        w_map = 1.0 if st == "fixed" else min(1.0, r["lidar"] / 0.6)
        before_H = self.shared.occ.entropy()
        self.shared.occ.update(pose, lid["angles"], lid["ranges"], lid["max_range"], w_map)
        info_gain = max(0.0, before_H - self.shared.occ.entropy())
        idx = self.shared.occ.last_idx
        if idx.size:
            m = np.zeros(self.shared.W * self.shared.H, bool)
            m[idx] = True
            self.shared.unc.observe(m.reshape(self.shared.H, self.shared.W), self.ekf.sigma_xy)
        w_rgb = 1.0 if st == "fixed" else min(1.0, r["rgb"] / 0.6)
        w_th = 1.0 if st == "fixed" else min(1.0, r["thermal"] / 0.6)
        geo_rgb = self.suite.sensors["rgb"].geo
        geo_th = self.suite.sensors["thermal"].geo
        self.shared.terrain.update_rgb(obs["rgb"].data["image"], geo_rgb, pose, w_rgb)
        self.shared.terrain.update_thermal(obs["thermal"].data["temp"], obs["thermal"].data["t_sat"], geo_th, pose, w_th)
        # victim candidates
        cl = {"rgb": detect_rgb(obs["rgb"].data["image"], geo_rgb) if "rgb" in self.modalities else [],
              "thermal": detect_thermal(obs["thermal"].data["temp"], obs["thermal"].data["t_sat"], geo_th)
              if "thermal" in self.modalities else [],
              "audio": detect_audio(obs["audio"].data) if "audio" in self.modalities else []}
        cw = {m: to_world(c, pose) for m, c in cl.items()}
        r_perc = {m: (r[m] if m in self.modalities else 0.0) for m in ("rgb", "thermal", "audio")}
        all_c = cw["rgb"] + cw["thermal"] + cw["audio"]
        newly = self.tracker.update(all_c, r_perc, self.t, lambda m, tx, ty: self._in_view(m, tx, ty, pose))
        usable = {m: h[m] > 0.5 and q[m] > 0.3 and m in self.modalities for m in ("rgb", "thermal", "audio")}
        self.cross.update_perception(cw, self.tracker.tracks, pose, usable)
        self.shared.update_victims(self.tracker.tracks)
        self.total_new_victims += len(newly)
        # logs (ground truth used ONLY for evaluation logs)
        self.est_traj.append(tuple(pose))
        self.true_traj.append((self.world.x, self.world.y, self.world.theta))
        tr = self.suite.true_reliability
        self.rel_log_est.append([r[s] for s in SENSOR_NAMES])
        self.rel_log_true.append([tr[s] for s in SENSOR_NAMES])
        self.last.update({"obs": obs, "rep": rep, "cands": cw, "cmd": cmd, "info_gain": info_gain,
                          "new_victims": len(newly), "quality_detail": {"rgb": rgbq, "thermal": thq},
                          "conditions": cond, "scan_match": sm})

    def _in_view(self, m, tx, ty, pose):
        x, y, th = pose
        c = self.cfg["sensors"][m]
        dx, dy = tx - x, ty - y
        d = math.hypot(dx, dy)
        if d > min(c["range"] - 0.5, 4.0) or d < 0.3:
            return False
        if abs(wrap(math.atan2(dy, dx) - th)) > math.radians(c["fov_deg"]) / 2 - 0.1:
            return False
        occ = self.shared.occ.occupied
        for s in np.linspace(0.1, 0.9, 8):
            cx, cy = int(x + s * dx), int(y + s * dy)
            if 0 <= cx < self.world.W and 0 <= cy < self.world.H and occ[cy, cx]:
                return False
        return True

    # ============================================================ evaluation
    def evaluate(self, match_radius: float = 2.0) -> dict:
        """Ground-truth evaluation metrics (simulation measurements)."""
        w = self.world
        est = np.array(self.est_traj)
        tru = np.array(self.true_traj)
        err = np.hypot(est[:, 0] - tru[:, 0], est[:, 1] - tru[:, 1])
        herr = np.abs(np.angle(np.exp(1j * (est[:, 2] - tru[:, 2]))))
        known = self.shared.occ.known
        cov = float((known & w.reachable).sum() / max(w.reachable.sum(), 1))
        est_occ = self.shared.occ.occupied
        occ_acc = float((est_occ[known] == w.grid[known]).mean()) if known.any() else 0.0
        # detection: greedy matching of confirmed tracks to GT victims
        conf = self.tracker.confirmed
        pairs = sorted(((math.hypot(t.x - v.x, t.y - v.y), i, j) for i, t in enumerate(conf)
                        for j, v in enumerate(w.victims)), key=lambda p: p[0])
        used_t, used_v, tdisc = set(), set(), []
        for d, i, j in pairs:
            if d <= match_radius and i not in used_t and j not in used_v:
                used_t.add(i)
                used_v.add(j)
                tdisc.append(conf[i].confirmed_t)
        tp = len(used_v)
        fp = len(conf) - len(used_t)
        nv = len(w.victims)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / nv if nv else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        re, rt = np.array(self.rel_log_est), np.array(self.rel_log_true)
        failed_cols = [i for i, s in enumerate(SENSOR_NAMES) if self.suite.degradation.failed(s)]
        fdr = float(np.mean(re[:, failed_cols] < self.cfg["reliability"]["failure_threshold"])) if failed_cols else float("nan")
        bins = np.clip((re.ravel() * 5).astype(int), 0, 4)
        ece = float(sum(abs(re.ravel()[bins == b].mean() - rt.ravel()[bins == b].mean()) * (bins == b).mean()
                        for b in range(5) if (bins == b).any()))
        step_ms = float(np.mean(self.step_times) * 1000) if self.step_times else 0.0
        return {
            "steps": self.t, "done_reason": self.done_reason,
            "ate": float(np.sqrt(np.mean(err ** 2))), "final_pos_error": float(err[-1]),
            "mean_heading_error": float(herr.mean()),
            "coverage": cov, "occupancy_accuracy": occ_acc,
            "victims_total": nv, "victims_found": tp, "false_positives": fp,
            "precision": prec, "recall": rec, "f1": f1,
            "mean_time_to_discovery": float(np.mean(tdisc)) if tdisc else float("nan"),
            "collisions": w.collisions, "hazard_exposure": float(w.hazard_exposure),
            "path_length": self.true_path_len, "safety_vetoes": self.safety.vetoes,
            "success": float(tp == nv),
            "efficiency": cov / max(self.true_path_len, 1.0) * 100,
            "rel_mae": float(np.mean(np.abs(re - rt))), "rel_ece": ece,
            "failure_detection_rate": fdr,
            "step_ms": step_ms,
        }
