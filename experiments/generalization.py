"""Experiment 6: generalisation to unseen maps across disaster severity and map difficulty.

  python -m experiments.generalization --seeds 4
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from experiments.exploration_comparison import methods
from experiments.metrics import EVAL_SEED_BASE, LOG_DIR, PLOT_DIR, line_plot, run_specs, summarize

SEVERITIES = (0.2, 0.5, 0.8)
DIFFICULTIES = ("easy", "hard")
METRICS = ["coverage", "victims_found", "f1", "ate", "collisions", "path_length"]
KEEP = ("E1 nearest", "E2 info-gain", "E2f info-gain/fixed", "E3 PPO/fixed", "E4 PPO/hybrid")


def specs(seeds):
    ms = {k: v for k, v in methods().items() if k in KEEP}
    out = []
    for sev in SEVERITIES:
        for dif in DIFFICULTIES:
            for d in ("earthquake", "fire", "flood"):
                for i in range(seeds):
                    for name, (rel, nav, pol) in ms.items():
                        s = {"severity": sev, "difficulty": dif, "disaster": d, "seed": EVAL_SEED_BASE + 900 + i,
                             "method": name, "reliability": rel, "nav": nav,
                             "degradation": {"lidar": "mild", "rgb": "mild"}}
                        if pol:
                            s["policy"] = pol
                        out.append(s)
    return out, list(ms)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--plots-only", action="store_true", help="re-draw plots from the saved CSV")
    a = ap.parse_args(argv)
    sp, order = specs(a.seeds)
    csv = os.path.join(LOG_DIR, "exp6_generalization.csv")
    df = pd.read_csv(csv) if a.plots_only else run_specs(sp, a.workers, csv)
    order = [m for m in order if m in set(df["method"])]
    summ = summarize(df, ["severity", "method"], METRICS)
    summ.to_csv(os.path.join(LOG_DIR, "exp6_summary_by_severity.csv"), index=False)
    summarize(df, ["difficulty", "method"], METRICS).to_csv(os.path.join(LOG_DIR, "exp6_summary_by_difficulty.csv"), index=False)
    for met, lab in (("coverage", "Area coverage"), ("f1", "Victim detection F1"), ("victims_found", "Victims found (of 5)")):
        line_plot(summ, "severity", "method", met, f"Exp 6 generalisation: {lab} vs disaster severity (unseen maps)",
                  lab, os.path.join(PLOT_DIR, f"exp6_{met}_vs_severity.png"), series_order=order)
    print(summ[["severity", "method"] + [f"{m}_mean" for m in METRICS]].round(3).to_string())


if __name__ == "__main__":
    main()
