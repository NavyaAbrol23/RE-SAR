"""B6 (optional): learned reliability calibrator.

A small PyTorch MLP maps the SAME interpretable signals used by the hybrid model
[environment prior E, observation quality Q, health H, cross-modal agreement A, sensor one-hot]
to the simulator's ground-truth effective reliability r* = 1 - e (supervised, simulation labels).
At run time the health gate is kept:  r = H_gate * MLP(features), so failures still give 0.
Inference uses a NumPy forward pass (no torch needed in the mission loop).

Train (uses CUDA automatically when available, e.g. on an RTX 4060):
  python -m reliability.learned --missions 60 --epochs 40 --device auto
Honest caveat: it learns the simulator's degradation model; it is a calibration baseline for the
hand-designed hybrid model, not evidence of real-world reliability estimation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sensors.base_sensor import SENSOR_NAMES

MODEL_PATH = os.path.join(ROOT, "rl", "models", "reliability_mlp.npz")
N_FEAT = 4 + len(SENSOR_NAMES)


def features(e, q, h, a, sensor):
    oh = [1.0 if s == sensor else 0.0 for s in SENSOR_NAMES]
    return [e, q, h, a] + oh


def model_available() -> bool:
    return os.path.exists(MODEL_PATH)


class LearnedReliability:
    _cache = None

    def __init__(self):
        if LearnedReliability._cache is None:
            d = np.load(MODEL_PATH)
            LearnedReliability._cache = [d[k] for k in ("W0", "b0", "W1", "b1", "W2", "b2")]
        self.W = LearnedReliability._cache

    def predict(self, X: np.ndarray) -> np.ndarray:
        W0, b0, W1, b1, W2, b2 = self.W
        h = np.maximum(X @ W0.T + b0, 0)
        h = np.maximum(h @ W1.T + b1, 0)
        return (1 / (1 + np.exp(-(h @ W2.T + b2)))).ravel()


# ------------------------------------------------------------------ training
def collect(n_missions: int, seed0: int = 5_000_000, max_steps: int = 250):
    from config.loader import load_config
    from core.mission import Mission
    rng = np.random.default_rng(0)
    X, Y = [], []
    for k in range(n_missions):
        d = ["earthquake", "fire", "flood"][k % 3]
        cfg = load_config(d, {"world": {"severity": float(rng.uniform(0.2, 0.9))}, "robot": {"max_steps": max_steps}})
        deg = {}
        for s in rng.choice(SENSOR_NAMES, size=int(rng.integers(0, 3)), replace=False):
            deg[str(s)] = str(rng.choice(["mild", "severe", "failed"]))
        m = Mission(cfg, seed=seed0 + k, reliability="hybrid", nav="infogain", degradation=deg)
        while not m.done:
            m.step()
            rep, tr = m.last["rep"], m.suite.true_reliability
            for s in SENSOR_NAMES:
                X.append(features(rep.env[s], rep.quality[s], rep.health[s], rep.agreement[s], s))
                Y.append(tr[s])
    return np.array(X, np.float32), np.array(Y, np.float32)


def train(n_missions=60, epochs=40, device="auto"):
    import torch
    import torch.nn as nn
    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" and torch.cuda.is_available() else "cpu"
    t0 = time.time()
    X, Y = collect(n_missions)
    t_collect = time.time() - t0
    idx = np.random.default_rng(1).permutation(len(X))
    cut = int(0.85 * len(X))
    tr, va = idx[:cut], idx[cut:]
    Xt, Yt = torch.tensor(X[tr], device=dev), torch.tensor(Y[tr], device=dev)
    Xv, Yv = torch.tensor(X[va], device=dev), torch.tensor(Y[va], device=dev)
    net = nn.Sequential(nn.Linear(N_FEAT, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    t1 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 1024):
            b = perm[i:i + 1024]
            loss = nn.functional.mse_loss(torch.sigmoid(net(Xt[b])).squeeze(1), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pv = torch.sigmoid(net(Xv)).squeeze(1)
        mae = float((pv - Yv).abs().mean())
        # baseline: the hand-designed hybrid formula evaluated on the same validation rows
        e, q, h, a = (Xv[:, i] for i in range(4))
        hyb = h * (0.35 * e + 0.45 * q + 0.2 * a) / 1.0
        mae_h = float((hyb - Yv).abs().mean())
    lin = [m for m in net if isinstance(m, nn.Linear)]
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    np.savez(MODEL_PATH, **{f"W{i}": l.weight.detach().cpu().numpy() for i, l in enumerate(lin)},
             **{f"b{i}": l.bias.detach().cpu().numpy() for i, l in enumerate(lin)})
    stats = {"device": dev, "samples": int(len(X)), "missions": n_missions, "epochs": epochs,
             "collect_seconds": round(t_collect, 1), "train_seconds": round(time.time() - t1, 1),
             "val_mae_learned": round(mae, 4), "val_mae_hybrid_formula_unsmoothed": round(mae_h, 4),
             "note": "validation rows are per-step samples from held-out shuffle of the same missions"}
    with open(os.path.join(ROOT, "results", "logs", "learned_reliability_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("[learned-reliability]", json.dumps(stats))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    train(a.missions, a.epochs, a.device)
