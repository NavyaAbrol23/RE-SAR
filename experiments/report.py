"""Build markdown tables from the executed experiment CSVs (used for docs/experimental_results.md).

  python -m experiments.report > results/logs/tables.md
"""
from __future__ import annotations

import json
import os

import pandas as pd

from experiments.metrics import LOG_DIR


def md(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    df = df.rename(columns=str)
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) and pd.notna(v) else ("–" if pd.isna(v) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def ci(mean, half):
    if pd.isna(mean):
        return "–"
    return f"{mean:.2f} ± {half:.2f}" if pd.notna(half) else f"{mean:.2f}"


def table_fixed_vs_hybrid(summary, exp, metrics):
    s = summary[summary.experiment == exp]
    rows = []
    for sc in dict.fromkeys(s.scenario):
        row = {"scenario": sc}
        for m in metrics:
            for st in ("fixed", "hybrid"):
                r = s[(s.scenario == sc) & (s.reliability == st)]
                row[f"{m} {st}"] = ci(r[f"{m}_mean"].iloc[0], r[f"{m}_ci95"].iloc[0]) if len(r) else "–"
        rows.append(row)
    return md(pd.DataFrame(rows))


def paired(path, filt=None):
    with open(path) as f:
        st = json.load(f)
    rows = []
    for r in st:
        if filt and not filt(r):
            continue
        if r.get("n", 0) == 0:
            continue
        rows.append({"experiment": r.get("experiment", ""), "A": r["a"], "B": r["b"], "metric": r["metric"], "n pairs": r["n"],
                     "mean(B−A)": f"{r['mean_diff_b_minus_a']:+.3f}",
                     "95% CI": f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]",
                     "B higher / tie / lower": f"{r['b_better']} / {r['tie']} / {r['b_worse']}"})
    return md(pd.DataFrame(rows))


def main():
    p = lambda f: os.path.join(LOG_DIR, f)
    if os.path.exists(p("exp123_summary.csv")):
        s = pd.read_csv(p("exp123_summary.csv"))
        for exp in ("exp1_individual", "exp2_failure", "exp3_simultaneous"):
            print(f"\n### {exp}\n")
            print(table_fixed_vs_hybrid(s, exp, ["f1", "ate", "coverage", "collisions"]))
            print()
            print(table_fixed_vs_hybrid(s, exp, ["rel_mae", "failure_detection_rate"]))
        print("\n### paired fixed→hybrid\n")
        print(paired(p("exp123_paired_stats.json")))
    if os.path.exists(p("exp4_summary_overall.csv")):
        o = pd.read_csv(p("exp4_summary_overall.csv"))
        cols = ["method"] + [c for c in o.columns if c.endswith("_mean") and c.split("_mean")[0] in
                             ("f1", "precision", "recall", "ate", "coverage", "collisions", "rel_mae", "rel_ece", "failure_detection_rate")]
        print("\n### exp4 overall\n")
        print(md(o[cols].rename(columns=lambda c: c.replace("_mean", ""))))
        b = pd.read_csv(p("exp4_summary_by_scenario.csv"))
        print("\n### exp4 F1 by scenario\n")
        print(md(b.pivot(index="method", columns="scenario", values="f1_mean").reset_index()))
        print("\n### exp4 ATE by scenario\n")
        print(md(b.pivot(index="method", columns="scenario", values="ate_mean").reset_index()))
        print("\n### exp4 paired vs fixed\n")
        print(paired(p("exp4_paired_stats.json")))
    if os.path.exists(p("exp5_summary_by_condition.csv")):
        o = pd.read_csv(p("exp5_summary.csv"))
        keep = ["coverage", "victims_found", "f1", "path_length", "steps", "collisions", "hazard_exposure", "efficiency", "success", "ate"]
        print("\n### exp5 overall\n")
        print(md(o[["method"] + [f"{k}_mean" for k in keep] + ["n"]].rename(columns=lambda c: c.replace("_mean", ""))))
        b = pd.read_csv(p("exp5_summary_by_condition.csv"))
        for m in ("coverage", "victims_found", "path_length"):
            print(f"\n### exp5 {m} by condition\n")
            print(md(b.pivot(index="method", columns="condition", values=f"{m}_mean").reset_index()))
        print("\n### exp5 paired\n")
        print(paired(p("exp5_paired_stats.json"), lambda r: r["a"] == "E2 info-gain"))
    if os.path.exists(p("exp6_summary_by_severity.csv")):
        s = pd.read_csv(p("exp6_summary_by_severity.csv"))
        for m in ("coverage", "victims_found", "f1"):
            print(f"\n### exp6 {m} by severity\n")
            print(md(s.pivot(index="method", columns="severity", values=f"{m}_mean").reset_index()))
        d = pd.read_csv(p("exp6_summary_by_difficulty.csv"))
        print("\n### exp6 coverage / victims by difficulty\n")
        print(md(d.pivot(index="method", columns="difficulty", values="coverage_mean").reset_index()))
        print(md(d.pivot(index="method", columns="difficulty", values="victims_found_mean").reset_index()))
    if os.path.exists(p("ppo_evaluation.csv")):
        e = pd.read_csv(p("ppo_evaluation.csv"))
        g = e.groupby("policy")[["episode_reward", "coverage", "victims_found", "f1", "path_length", "steps"]].mean().reset_index()
        print("\n### rl.evaluate\n")
        print(md(g))
    for f in ("computational.json", "learned_reliability_stats.json"):
        if os.path.exists(p(f)):
            print(f"\n### {f}\n```\n{open(p(f)).read()}\n```")


if __name__ == "__main__":
    main()
