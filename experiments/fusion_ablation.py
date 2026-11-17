"""Experiment 4: fusion ablation (B1-B5, + optional B6 learned) and B0 single-modality baselines.

  B0  rgb_only / thermal_only / audio_only  (victim perception from one modality, hybrid weights)
  B1  fixed                 B2 env              B3 quality
  B4  env_quality           B4+ env_quality_health
  B5  hybrid (env + quality + health + cross-modal)
  B6  learned  (PyTorch MLP calibrator; only if reliability/learned model exists)

  python -m experiments.fusion_ablation --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from experiments.metrics import (EVAL_SEED_BASE, LOG_DIR, PLOT_DIR, category_bars, grouped_bars, paired_diff, run_specs,
                                 summarize)
from reliability.learned import model_available

DISASTERS = ("earthquake", "fire", "flood")
SCENARIOS = {
    "nominal": {},
    "thermal_failed": {"thermal": "failed"},
    "rgb_failed+audio_severe": {"rgb": "failed", "audio": "severe"},
    "odometry_failed+lidar_mild": {"odometry": "failed", "lidar": "mild"},
    "mixed_severe": {"rgb": "severe", "lidar": "severe", "thermal": "severe"},
}
STRATS = ["fixed", "env", "quality", "env_quality", "env_quality_health", "hybrid"]
B0 = {"rgb_only": ("rgb",), "thermal_only": ("thermal",), "audio_only": ("audio",)}
GROUPS = {"rgb_only": "B0", "thermal_only": "B0", "audio_only": "B0", "fixed": "B1", "env": "B2", "quality": "B3",
          "env_quality": "B4", "env_quality_health": "B4+H", "hybrid": "B5", "learned": "B6"}
METRICS = ["f1", "precision", "recall", "ate", "coverage", "collisions", "rel_mae", "rel_ece",
           "failure_detection_rate", "victims_found", "false_positives"]


def specs(seeds, include_learned):
    strats = STRATS + (["learned"] if include_learned else [])
    out = []
    for sc, deg in SCENARIOS.items():
        for d in DISASTERS:
            for i in range(seeds):
                base = {"scenario": sc, "disaster": d, "seed": EVAL_SEED_BASE + 100 + i, "nav": "infogain", "degradation": deg}
                for st in strats:
                    out.append({**base, "method": st, "reliability": st})
                for name, mods in B0.items():
                    out.append({**base, "method": name, "reliability": "hybrid", "modalities": mods})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--plots-only", action="store_true", help="re-draw from the saved CSV")
    a = ap.parse_args(argv)
    csv = os.path.join(LOG_DIR, "exp4_fusion_ablation.csv")
    if a.plots_only:
        df = pd.read_csv(csv)
        learned = "learned" in set(df["method"])
    else:
        learned = model_available()
        df = run_specs(specs(a.seeds, learned), a.workers, csv)
    order = list(B0) + STRATS + (["learned"] if learned else [])
    overall = summarize(df, ["method"], METRICS)
    overall["scenario"] = "all scenarios"
    by_sc = summarize(df, ["scenario", "method"], METRICS)
    overall.to_csv(os.path.join(LOG_DIR, "exp4_summary_overall.csv"), index=False)
    by_sc.to_csv(os.path.join(LOG_DIR, "exp4_summary_by_scenario.csv"), index=False)
    for met, lab in (("f1", "Victim detection F1"), ("ate", "Localisation ATE (cells)"),
                     ("rel_mae", "Reliability MAE vs simulator truth"), ("collisions", "Collisions per mission")):
        category_bars(overall, "method", met, f"Exp 4 fusion ablation: {lab} (all scenarios, mean ± 95% CI)", lab,
                      os.path.join(PLOT_DIR, f"exp4_ablation_{met}.png"), order=order,
                      highlight=("hybrid", "env_quality_health", "learned"), groups=GROUPS)
    grouped_bars(by_sc[by_sc.method.isin(["fixed", "env_quality", "env_quality_health", "hybrid"])], "scenario", "method",
                 "f1", "Exp 4: detection F1 by scenario (selected strategies)", "F1",
                 os.path.join(PLOT_DIR, "exp4_f1_by_scenario.png"),
                 series_order=["fixed", "env_quality", "env_quality_health", "hybrid"], ylim=(0, 1.1))
    stats = [paired_diff(df, ["scenario", "disaster", "seed"], "method", "fixed", m, met)
             for m in order if m != "fixed" for met in ("f1", "ate")]
    with open(os.path.join(LOG_DIR, "exp4_paired_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    cols = ["method"] + [f"{m}_mean" for m in ("f1", "precision", "recall", "ate", "coverage", "collisions", "rel_mae", "failure_detection_rate")]
    print(overall.set_index("method").loc[order].reset_index()[cols].round(3).to_string())


if __name__ == "__main__":
    main()
