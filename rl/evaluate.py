"""Evaluate trained PPO policies on UNSEEN randomized maps, against heuristic exploration.

Each episode uses the same (disaster, map seed, severity, degradation) for every policy.
Reports RL episode reward plus mission metrics (coverage, victims found, F1, path length, ...).

  python -m rl.evaluate --episodes 12
  python -m rl.evaluate --episodes 12 --model rl/models/ppo_hybrid.zip --reliability hybrid
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from navigation.frontier import select_infogain
from rl.observation import K_FRONTIERS
from rl.policy import default_model_path
from rl.sar_env import EVAL_SEED_BASE, SAREnvironment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def episode_options(n, seed=123):
    rng = np.random.default_rng(seed)
    opts = []
    for i in range(n):
        deg = {}
        if rng.random() < 0.6:
            s = str(rng.choice(["rgb", "thermal", "lidar", "imu", "odometry", "audio"]))
            deg[s] = str(rng.choice(["mild", "severe", "failed"]))
        opts.append({"disaster": ["earthquake", "fire", "flood"][i % 3], "severity": float(rng.uniform(0.2, 0.8)),
                     "map_seed": EVAL_SEED_BASE + 2000 + i, "degradation": deg})
    return opts


def run(policy_name, act_fn, reliability, opts):
    env = SAREnvironment(reliability=reliability)
    rows = []
    for o in opts:
        obs, _ = env.reset(seed=0, options=o)
        R, done = 0.0, False
        while not done:
            a = act_fn(env, obs)
            obs, r, term, trunc, _ = env.step(a)
            R += r
            done = term or trunc
        ev = env.mission.evaluate()
        rows.append({"policy": policy_name, "reliability": reliability, "disaster": o["disaster"],
                     "map_seed": o["map_seed"], "degradation": ",".join(f"{k}={v}" for k, v in o["degradation"].items()) or "none",
                     "episode_reward": R, **{k: ev[k] for k in ("coverage", "victims_found", "f1", "path_length",
                                                               "steps", "collisions", "ate", "done_reason")}})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--model", default=None, help="evaluate only this model")
    ap.add_argument("--reliability", default="hybrid", choices=["fixed", "hybrid"])
    a = ap.parse_args(argv)
    from stable_baselines3 import PPO
    opts = episode_options(a.episodes)
    rows = []
    models = [(a.model, a.reliability)] if a.model else [(default_model_path(r), r) for r in ("hybrid", "fixed")]
    for path, rel in models:
        if not os.path.exists(path):
            print(f"[evaluate] missing {path} - train it with: python -m rl.train --reliability {rel}")
            continue
        model = PPO.load(path, device="cpu")
        rows += run(f"PPO/{rel}", lambda env, obs, m=model: int(m.predict(obs, deterministic=True)[0]), rel, opts)
    rows += run("nearest/hybrid", lambda env, obs: 0, "hybrid", opts)
    rows += run("infogain/hybrid", lambda env, obs: select_infogain(env.mission.frontiers, env.cfg) or 0, "hybrid", opts)
    rows += run("random/hybrid", lambda env, obs: int(env.np_random.integers(max(1, len(env.mission.frontiers)))), "hybrid", opts)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "logs", "ppo_evaluation.csv")
    df.to_csv(out, index=False)
    summ = df.groupby("policy")[["episode_reward", "coverage", "victims_found", "f1", "path_length", "steps", "collisions"]].agg(["mean", "std"])
    print(summ.round(3).to_string())
    print(f"[evaluate] {len(df)} episodes -> {out}")
    return df


if __name__ == "__main__":
    main()
