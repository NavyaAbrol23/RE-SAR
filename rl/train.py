"""Train PPO (Stable-Baselines3) on SAREnvironment.

Examples
  python -m rl.train --reliability hybrid --timesteps 100000            # E4: PPO + hybrid reliability
  python -m rl.train --reliability fixed  --timesteps 100000            # E3: PPO + fixed weights
  python -m rl.train --reliability hybrid --timesteps 2048 --n-envs 1   # smoke test
Device: --device auto picks CUDA (e.g. RTX 4060) when PyTorch sees a GPU, otherwise CPU.
Note: with a small MlpPolicy, simulation (CPU) dominates run time; the GPU mainly speeds updates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config.loader import load_config
from rl.sar_env import SAREnvironment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but not available -> falling back to CPU")
        return "cpu"
    return name


def make_env(reliability, rank, log_dir, seed):
    def _f():
        env = SAREnvironment(reliability=reliability, seed=seed + rank)
        return Monitor(env, os.path.join(log_dir, f"monitor_{rank}"),
                       info_keywords=("confirmed", "disaster"))
    return _f


def plot_curve(log_dir, out_png, title):
    frames = []
    for f in os.listdir(log_dir):
        if f.startswith("monitor_") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(log_dir, f), skiprows=1)
            frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames).sort_values("t").reset_index(drop=True)
    df["steps"] = df["l"].cumsum()
    roll = df["r"].rolling(max(5, len(df) // 20), min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["steps"], df["r"], ".", alpha=0.25, color="#8aa", label="episode reward")
    ax.plot(df["steps"], roll, color="#145", lw=2, label="rolling mean")
    ax.set_xlabel("environment steps (RL decisions)")
    ax.set_ylabel("episode reward")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reliability", default="hybrid", choices=["fixed", "hybrid"])
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    cfg = load_config("earthquake")
    p = cfg["rl"]["ppo"]
    total = a.timesteps or int(p["total_timesteps"])
    device = pick_device(a.device)
    name = f"ppo_{a.reliability}{a.tag}"
    log_dir = os.path.join(ROOT, "results", "logs", name)
    ck_dir = os.path.join(ROOT, "results", "checkpoints", name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ck_dir, exist_ok=True)
    for f in os.listdir(log_dir):
        if f.startswith("monitor_"):
            os.remove(os.path.join(log_dir, f))
    fns = [make_env(a.reliability, i, log_dir, a.seed * 1000) for i in range(a.n_envs)]
    venv = SubprocVecEnv(fns) if a.n_envs > 1 else DummyVecEnv(fns)
    n_steps = max(64, min(int(p["n_steps"]), total // a.n_envs))
    model = PPO("MlpPolicy", venv, learning_rate=p["learning_rate"], n_steps=n_steps,
                batch_size=min(int(p["batch_size"]), n_steps * a.n_envs), gamma=p["gamma"],
                gae_lambda=p["gae_lambda"], clip_range=p["clip_range"], verbose=1, seed=a.seed,
                device=device, policy_kwargs={"net_arch": [256, 128]})
    print(f"[train] {name}: {total} timesteps, {a.n_envs} envs, device={device}, "
          f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    t0 = time.time()
    model.learn(total_timesteps=total, callback=CheckpointCallback(max(total // 5, 1000) // a.n_envs, ck_dir, name))
    dur = time.time() - t0
    os.makedirs(os.path.join(ROOT, "rl", "models"), exist_ok=True)
    model_path = os.path.join(ROOT, "rl", "models", f"{name}.zip")
    model.save(model_path)
    model.save(os.path.join(ck_dir, f"{name}_final.zip"))
    venv.close()
    df = plot_curve(log_dir, os.path.join(ROOT, "results", "plots", f"reward_curve_{name}.png"),
                    f"PPO training reward ({a.reliability} reliability)")
    stats = {"name": name, "timesteps": total, "n_envs": a.n_envs, "device": device,
             "training_seconds": round(dur, 1), "episodes": int(len(df)) if df is not None else 0,
             "final_rolling_reward": float(df["r"].tail(max(5, len(df) // 10)).mean()) if df is not None and len(df) else None,
             "model_path": os.path.relpath(model_path, ROOT)}
    with open(os.path.join(log_dir, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[train] done:", json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
