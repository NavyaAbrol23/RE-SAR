"""Experiments 1-3: individual degradation, complete single-sensor failure, simultaneous degradation.
Fixed-weight fusion (B1) vs hybrid reliability (B5), info-gain exploration, paired seeds.

  python -m experiments.sensor_failure --seeds 4
"""
from __future__ import annotations

import argparse
import json
import os

from experiments.metrics import (EVAL_SEED_BASE, LOG_DIR, PLOT_DIR, grouped_bars, paired_diff, run_specs,
                                 summarize)
from sensors.base_sensor import SENSOR_NAMES

DISASTERS = ("earthquake", "fire", "flood")
EXP1 = {f"{s}_severe": {s: "severe"} for s in SENSOR_NAMES}  # rgb visibility, thermal sat., lidar returns, imu drift, odo slip, audio noise
EXP2 = {f"{s}_failed": {s: "failed"} for s in SENSOR_NAMES}
EXP3 = {  # scenario -> (disasters, degradation)
    "fire:rgb+thermal": (("fire",), {"rgb": "severe", "thermal": "severe"}),
    "earthquake:lidar+odometry": (("earthquake",), {"lidar": "severe", "odometry": "severe"}),
    "flood:lidar+odometry+audio": (("flood",), {"lidar": "severe", "odometry": "severe", "audio": "severe"}),
    "mixed:rgb+lidar+thermal": (DISASTERS, {"rgb": "severe", "lidar": "severe", "thermal": "severe"}),
}
METRICS = ["f1", "precision", "recall", "ate", "coverage", "collisions", "hazard_exposure", "rel_mae",
           "failure_detection_rate", "victims_found", "occupancy_accuracy"]
STRATS = ("fixed", "hybrid")


def specs(seeds):
    out = []
    for exp, scen in (("exp1_individual", {"nominal": {}, **EXP1}), ("exp2_failure", EXP2)):
        for name, deg in scen.items():
            for d in DISASTERS:
                for i in range(seeds):
                    for st in STRATS:
                        out.append({"experiment": exp, "scenario": name, "disaster": d, "seed": EVAL_SEED_BASE + i,
                                    "reliability": st, "nav": "infogain", "degradation": deg})
    for name, (dis, deg) in EXP3.items():
        for d in dis:
            for i in range(seeds):
                for st in STRATS:
                    out.append({"experiment": "exp3_simultaneous", "scenario": name, "disaster": d,
                                "seed": EVAL_SEED_BASE + i, "reliability": st, "nav": "infogain", "degradation": deg})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    df = run_specs(specs(a.seeds), a.workers, os.path.join(LOG_DIR, "exp123_sensor_failure.csv"))
    summ = summarize(df, ["experiment", "scenario", "reliability"], METRICS)
    summ.to_csv(os.path.join(LOG_DIR, "exp123_summary.csv"), index=False)
    stats = []
    for exp in df["experiment"].unique():
        sub = df[df.experiment == exp]
        for met in ("f1", "ate", "coverage", "collisions", "rel_mae"):
            r = paired_diff(sub, ["scenario", "disaster", "seed"], "reliability", "fixed", "hybrid", met)
            r["experiment"] = exp
            stats.append(r)
        s = summ[summ.experiment == exp]
        tag = exp.split("_")[0]
        for met, lab, yl in (("f1", "Victim detection F1", (0, 1.1)), ("ate", "Localisation ATE (cells)", None),
                             ("coverage", "Area coverage", (0, 1.1)), ("rel_mae", "Reliability-estimate MAE vs simulator truth", None)):
            grouped_bars(s, "scenario", "reliability", met, f"{exp}: {lab} (fixed vs hybrid, mean ± 95% CI)", lab,
                         os.path.join(PLOT_DIR, f"{tag}_{met}.png"), series_order=list(STRATS), ylim=yl)
    with open(os.path.join(LOG_DIR, "exp123_paired_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(summ[["experiment", "scenario", "reliability", "f1_mean", "ate_mean", "coverage_mean", "collisions_mean",
                "rel_mae_mean", "failure_detection_rate_mean"]].round(3).to_string())
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()
