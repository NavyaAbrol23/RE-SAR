"""Shared experiment machinery: run missions (in parallel), aggregate, paired statistics, plots.

All numbers are SIMULATION measurements from executed missions. Each method is run on exactly the
same (disaster, map seed, degradation) tuples, so comparisons are paired.
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.loader import load_config
from core.mission import Mission

LOG_DIR = os.path.join(ROOT, "results", "logs")
PLOT_DIR = os.path.join(ROOT, "results", "plots")
EVAL_SEED_BASE = 10_000_000  # disjoint from PPO training seeds -> unseen maps

# validated categorical palette (dataviz reference instance, light mode), fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"

# colour follows the entity, never its position in a particular chart
ENTITY_COLORS = {
    "fixed": SERIES[0], "hybrid": SERIES[1],
    "E0 random": SERIES[0], "E1 nearest": SERIES[1], "E2 info-gain": SERIES[2], "E2f info-gain/fixed": SERIES[3],
    "E3 PPO/fixed": SERIES[4], "E4 PPO/hybrid": SERIES[5],
    "env_quality": SERIES[3], "env_quality_health": SERIES[6],
}


def color_for(name, i):
    return ENTITY_COLORS.get(name, SERIES[i % len(SERIES)])


_CHOOSERS = {}


def _chooser(kind):
    if kind not in _CHOOSERS:
        from rl.policy import try_load
        _CHOOSERS[kind] = try_load(kind)
    return _CHOOSERS[kind]


class _Timed:
    def __init__(self, ch):
        self.ch, self.times = ch, []

    def __call__(self, m):
        t = time.perf_counter()
        a = self.ch(m)
        self.times.append(time.perf_counter() - t)
        return a


def run_mission(spec: dict) -> dict:
    """spec keys: disaster, seed, reliability, nav, degradation, [modalities, severity, difficulty,
    policy ('fixed'|'hybrid' PPO model), label, plus any tag columns]."""
    ov = {"world": {}}
    if "severity" in spec:
        ov["world"]["severity"] = spec["severity"]
    if "difficulty" in spec:
        ov["world"]["difficulty"] = spec["difficulty"]
    cfg = load_config(spec["disaster"], ov)
    chooser, nav = None, spec["nav"]
    if nav == "ppo":
        base = _chooser(spec.get("policy", "hybrid"))
        if base is None:
            return {**_tags(spec), "error": "no_model"}
        chooser = _Timed(base)
    m = Mission(cfg, seed=spec["seed"], reliability=spec["reliability"], nav=nav,
                degradation=spec.get("degradation") or {},
                modalities=tuple(spec.get("modalities", ("rgb", "thermal", "audio"))), chooser=chooser)
    t0 = time.perf_counter()
    while not m.done:
        m.step()
    wall = time.perf_counter() - t0
    ev = m.evaluate()
    ev.update(_tags(spec))
    ev["wall_s"] = wall
    ev["steps_per_s"] = m.t / max(wall, 1e-9)
    ev["inference_ms"] = float(np.mean(chooser.times) * 1000) if chooser and chooser.times else np.nan
    return ev


def _tags(spec):
    out = {k: v for k, v in spec.items() if k not in ("degradation", "modalities")}
    out["degradation"] = ",".join(f"{k}={v}" for k, v in sorted((spec.get("degradation") or {}).items())) or "none"
    out["modalities"] = "+".join(spec.get("modalities", ("rgb", "thermal", "audio")))
    return out


def run_specs(specs, workers: int | None = None, out_csv: str | None = None) -> pd.DataFrame:
    workers = workers or max(1, (os.cpu_count() or 2))
    t0 = time.time()
    if workers > 1 and len(specs) > 1:
        with Pool(workers) as p:
            rows = p.map(run_mission, specs, chunksize=1)
    else:
        rows = [run_mission(s) for s in specs]
    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
    print(f"[experiments] {len(specs)} missions in {time.time() - t0:.0f}s -> {out_csv}")
    return df


def summarize(df: pd.DataFrame, by, metrics) -> pd.DataFrame:
    g = df.groupby(by)
    out = g[metrics].mean().add_suffix("_mean")
    sd = g[metrics].std().add_suffix("_std")
    n = g.size().rename("n")
    ci = (1.96 * g[metrics].std() / np.sqrt(g.size().values[:, None])).add_suffix("_ci95")
    return pd.concat([out, sd, ci, n], axis=1).reset_index()


def paired_diff(df, pair_cols, method_col, a, b, metric, n_boot=4000, seed=0):
    """Mean of (b - a) over paired runs with a bootstrap 95% CI and win/tie/loss counts."""
    A = df[df[method_col] == a].set_index(pair_cols)[metric]
    B = df[df[method_col] == b].set_index(pair_cols)[metric]
    j = A.index.intersection(B.index)
    d = (B.loc[j] - A.loc[j]).astype(float).dropna().values
    if len(d) == 0:
        return {"metric": metric, "a": a, "b": b, "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    return {"metric": metric, "a": a, "b": b, "n": len(d), "mean_diff_b_minus_a": float(d.mean()),
            "ci95_low": float(np.percentile(boots, 2.5)), "ci95_high": float(np.percentile(boots, 97.5)),
            "b_better": int((d > 1e-9).sum()), "tie": int((np.abs(d) <= 1e-9).sum()), "b_worse": int((d < -1e-9).sum())}


# -------------------------------------------------------------------- plotting
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def grouped_bars(summary: pd.DataFrame, group_col, series_col, metric, title, ylabel, path,
                 series_order=None, group_order=None, ylim=None, fmt="{:.2f}"):
    """Grouped bar chart with 95% CI whiskers, value labels and a legend (identity never color-only)."""
    groups = group_order or list(dict.fromkeys(summary[group_col]))
    series = series_order or list(dict.fromkeys(summary[series_col]))
    n = len(series)
    width = min(0.8 / n, 0.28)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(groups) * (1 + n * 0.25)), 4.2), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        vals, errs = [], []
        for g in groups:
            r = summary[(summary[group_col] == g) & (summary[series_col] == s)]
            vals.append(float(r[f"{metric}_mean"].iloc[0]) if len(r) else np.nan)
            errs.append(float(r[f"{metric}_ci95"].iloc[0]) if len(r) else 0)
        pos = x + (i - (n - 1) / 2) * (width + 0.02)
        bars = ax.bar(pos, vals, width, color=color_for(s, i), label=s, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(pos, vals, yerr=np.nan_to_num(errs), fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
        if n <= 4:
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center",
                            va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, groups, rotation=20 if len(groups) > 5 else 0, ha="right" if len(groups) > 5 else "center")
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8, ncol=min(n, 4), loc="upper left", bbox_to_anchor=(0, -0.18 if len(groups) > 5 else -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def line_plot(summary, x_col, series_col, metric, title, ylabel, path, series_order=None):
    series = series_order or list(dict.fromkeys(summary[series_col]))
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    _style(ax)
    xs = sorted(summary[x_col].unique())
    span = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
    for i, s in enumerate(series):
        r = summary[summary[series_col] == s].sort_values(x_col)
        dx = (i - (len(series) - 1) / 2) * 0.012 * span  # small dodge so CI whiskers do not overlap
        c = color_for(s, i)
        ax.errorbar(r[x_col] + dx, r[f"{metric}_mean"], yerr=r[f"{metric}_ci95"], color=c, lw=2, marker="o", ms=7,
                    capsize=2, elinewidth=1, ecolor=c, alpha=0.95, label=s, mec=SURFACE, mew=1.5)
    ax.set_xticks(xs)
    ax.set_xlabel(x_col, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.14))
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def category_bars(summary: pd.DataFrame, cat_col, metric, title, xlabel, path, order=None, highlight=(),
                  fmt="{:.2f}", groups=None):
    """Horizontal bars, one category per row (identity carried by the axis label, not by colour).
    `highlight` categories use series-1; others a muted neutral. `groups` = {category: group label}."""
    cats = order or list(summary[cat_col])
    vals = [float(summary.loc[summary[cat_col] == c, f"{metric}_mean"].iloc[0]) for c in cats]
    errs = [float(summary.loc[summary[cat_col] == c, f"{metric}_ci95"].iloc[0]) for c in cats]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(cats) + 1.3), facecolor=SURFACE)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    y = np.arange(len(cats))[::-1]
    cols = [SERIES[0] if c in highlight else "#a9b4c2" for c in cats]
    ax.barh(y, vals, 0.62, color=cols, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(vals, y, xerr=np.nan_to_num(errs), fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + (e if np.isfinite(e) else 0) + 0.01 * max(max(vals), 1e-6), yi, fmt.format(v), va="center",
                fontsize=8, color=INK)
    labels = [f"{c}  ({groups[c]})" if groups and c in groups else c for c in cats]
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path
