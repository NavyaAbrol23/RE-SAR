"""Experiment 5: exploration comparison on UNSEEN maps (seeds disjoint from PPO training).

  E0 random   E1 nearest   E2 info-gain   (heuristics; hybrid reliability)
  E2f info-gain with fixed reliability (reference)
  E3 PPO + fixed reliability   E4 PPO + hybrid reliability
Also records computational metrics (steps/s, PPO inference latency, peak Python memory, training time).

  python -m experiments.exploration_comparison --seeds 6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import tracemalloc

import numpy as np
import pandas as pd

from experiments.metrics import (EVAL_SEED_BASE, LOG_DIR, PLOT_DIR, grouped_bars, paired_diff, run_mission,
                                 run_specs, summarize)
from rl.policy import default_model_path

DISASTERS = ("earthquake", "fire", "flood")
CONDITIONS = {"nominal": {}, "degraded": {"rgb": "severe", "odometry": "failed", "audio": "mild"},
              "lidar+thermal": {"lidar": "severe", "thermal": "failed"}}
METRICS = ["coverage", "victims_found", "recall", "f1", "path_length", "steps", "collisions", "hazard_exposure",
           "efficiency", "mean_time_to_discovery", "success", "ate"]


def methods():
    m = {"E0 random": ("hybrid", "random", None), "E1 nearest": ("hybrid", "nearest", None),
         "E2 info-gain": ("hybrid", "infogain", None), "E2f info-gain/fixed": ("fixed", "infogain", None)}
    if os.path.exists(default_model_path("fixed")):
        m["E3 PPO/fixed"] = ("fixed", "ppo", "fixed")
    if os.path.exists(default_model_path("hybrid")):
        m["E4 PPO/hybrid"] = ("hybrid", "ppo", "hybrid")
    return m


def specs(seeds):
    out = []
    for cond, deg in CONDITIONS.items():
        for d in DISASTERS:
            for i in range(seeds):
                for name, (rel, nav, pol) in methods().items():
                    s = {"condition": cond, "disaster": d, "seed": EVAL_SEED_BASE + 500 + i, "method": name,
                         "reliability": rel, "nav": nav, "degradation": deg}
                    if pol:
                        s["policy"] = pol
                    out.append(s)
    return out


def computational():
    res = {}
    spec = {"disaster": "fire", "seed": EVAL_SEED_BASE, "reliability": "hybrid", "nav": "infogain", "degradation": {}}
    tracemalloc.start()  # memory run (tracemalloc slows execution, so timing is measured separately)
    run_mission(spec)
    res["peak_python_memory_MB_one_mission"] = round(tracemalloc.get_traced_memory()[1] / 1e6, 1)
    tracemalloc.stop()
    ev = run_mission(spec)
    res["mission_steps_per_s"] = round(ev["steps_per_s"], 1)
    res["perception_action_step_ms"] = round(ev["step_ms"], 2)
    for f in glob.glob(os.path.join(LOG_DIR, "ppo_*", "training_stats.json")):
        with open(f) as fh:
            st = json.load(fh)
        res[f"training_{st['name']}"] = {k: st[k] for k in ("timesteps", "device", "training_seconds", "episodes")}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--plots-only", action="store_true", help="re-draw plots from the saved CSV")
    a = ap.parse_args(argv)
    csv = os.path.join(LOG_DIR, "exp5_exploration.csv")
    df = pd.read_csv(csv) if a.plots_only else run_specs(specs(a.seeds), a.workers, csv)
    order = [m for m in methods() if m in set(df["method"])]
    summ = summarize(df, ["method"], METRICS)
    summ["all"] = "all conditions"
    summ.to_csv(os.path.join(LOG_DIR, "exp5_summary.csv"), index=False)
    by_c = summarize(df, ["condition", "method"], METRICS)
    by_c.to_csv(os.path.join(LOG_DIR, "exp5_summary_by_condition.csv"), index=False)
    for met, lab, yl in (("coverage", "Area coverage (unseen maps)", (0, 1.1)), ("victims_found", "Victims found (of 5)", (0, 5.5)),
                         ("path_length", "Path length (cells)", None), ("efficiency", "Coverage per 100 cells travelled", None)):
        grouped_bars(by_c, "condition", "method", met, f"Exp 5 exploration: {lab} (mean ± 95% CI)", lab,
                     os.path.join(PLOT_DIR, f"exp5_{met}.png"), series_order=order, ylim=yl)
    if a.plots_only:
        return
    stats = []
    for ref in ("E2 info-gain", "E1 nearest"):
        for m in order:
            if m != ref:
                for met in ("coverage", "victims_found", "path_length"):
                    stats.append(paired_diff(df, ["condition", "disaster", "seed"], "method", ref, m, met))
    comp = computational()
    comp["ppo_inference_ms_mean"] = {m: round(float(np.nanmean(df[df.method == m]["inference_ms"])), 3)
                                     for m in order if "PPO" in m}
    with open(os.path.join(LOG_DIR, "exp5_paired_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    with open(os.path.join(LOG_DIR, "computational.json"), "w") as f:
        json.dump(comp, f, indent=1)
    print(summ.set_index("method").loc[order][[f"{m}_mean" for m in METRICS]].round(3).to_string())
    print(json.dumps(comp, indent=1))


if __name__ == "__main__":
    main()
