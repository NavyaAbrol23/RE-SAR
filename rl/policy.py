"""Load a trained PPO model and use it as a frontier chooser inside a Mission (GUI / experiments)."""
from __future__ import annotations

import os

import numpy as np

from rl.observation import K_FRONTIERS, build_observation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_model_path(reliability: str = "hybrid") -> str:
    return os.path.join(ROOT, "rl", "models", f"ppo_{reliability}.zip")


class PPOChooser:
    def __init__(self, path: str, device: str = "cpu"):
        from stable_baselines3 import PPO  # imported lazily so the app runs without a model
        self.model = PPO.load(path, device=device)
        self.path = path
        self.last_action = None

    def __call__(self, mission) -> int:
        obs = build_observation(mission)
        a, _ = self.model.predict(obs, deterministic=True)
        a = int(a)
        self.last_action = a
        if a >= len(mission.frontiers):  # STOP or empty slot -> nearest frontier (mission keeps going)
            return 0
        return a


def try_load(reliability: str = "hybrid", path: str | None = None):
    p = path or default_model_path(reliability)
    if not os.path.exists(p):
        return None
    try:
        return PPOChooser(p)
    except Exception as e:  # corrupt / incompatible model: fall back to heuristics
        print(f"[policy] could not load {p}: {e}")
        return None
