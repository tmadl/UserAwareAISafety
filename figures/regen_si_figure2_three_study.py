#!/usr/bin/env python3
"""Regenerate si_figure2_three_study_replication using the headline scorer
(Qwen3.5-ORPO-400 logit-EV) and the canonical z(IC)^2 parameterisation.

Replaces the stale version whose panel annotations came from gpt-4.1-mini
scoring without the canonical covariates. Panel betas reproduce the SI
caption: Study 1 -1.89 (p=.056, N=495), Study 2 -1.96 (p=.005, N=577),
Study 3 -2.44 (p<.001, N=710).

Output: paper/PNAS_Nexus_v3/figures/si_figure2_three_study_replication.{pdf,png}
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "c", HERE.parent / "analysis" / "01_costello_analysis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

OUT = Path("/mnt/workvm/UserAwareAISafety/paper/PNAS_Nexus_v3/figures")

df = mod.load_data()
df = df.loc[:, ~df.columns.duplicated()]
df = df.dropna(subset=["IC_q400_logit", "DV_BeliefChange_Specific",
                       "Pre_Belief_Specific", "OpenendedResponseWordCount",
                       "StudyNumber"]).copy()


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


study_colors = {1: "#2166AC", 2: "#4393C3", 3: "#92C5DE"}

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.6))

for i_s, (ax, s) in enumerate(zip(axes, [1, 2, 3])):
    sub = df[df["StudyNumber"] == s].copy()
    ic_z = zs(sub["IC_q400_logit"].values)
    y = sub["DV_BeliefChange_Specific"].values
    pb_z = zs(sub["Pre_Belief_Specific"].values)
    wc_z = zs(sub["OpenendedResponseWordCount"].values)
    color = study_colors[s]

    X = sm.add_constant(np.column_stack([ic_z, ic_z**2, pb_z, wc_z]))
    m = sm.OLS(y, X).fit()
    beta_ic2, p_ic2 = m.params[2], m.pvalues[2]

    jx = ic_z + np.random.RandomState(i_s).normal(0, 0.03, len(ic_z))
    jy = y + np.random.RandomState(i_s + 10).normal(0, 0.3, len(y))
    ax.scatter(jx, jy, s=3, alpha=0.08, color=color, edgecolors="none",
               rasterized=True)

    xgrid = np.linspace(ic_z.min(), ic_z.max(), 200)
    preds = np.zeros((500, 200))
    rng = np.random.RandomState(42)
    for b in range(500):
        idx = rng.choice(len(ic_z), len(ic_z), replace=True)
        c = np.polyfit(ic_z[idx], y[idx], 2)
        preds[b] = np.polyval(c, xgrid)
    lo = np.nanpercentile(preds, 2.5, axis=0)
    hi = np.nanpercentile(preds, 97.5, axis=0)
    med = np.nanpercentile(preds, 50, axis=0)
    ax.fill_between(xgrid, lo, hi, alpha=0.2, color=color, linewidth=0)
    ax.plot(xgrid, med, color=color, linewidth=1.5)

    sub["_ic_z"] = ic_z
    sub["_q"] = pd.qcut(sub["_ic_z"].rank(method="first"), 5, labels=False)
    for q in range(5):
        sq = sub[sub["_q"] == q]["DV_BeliefChange_Specific"]
        xm = sub[sub["_q"] == q]["_ic_z"].mean()
        ci = 1.96 * sq.std() / np.sqrt(len(sq))
        ax.errorbar(xm, sq.mean(), yerr=ci, fmt="o", color=color,
                    markersize=4, markerfacecolor="white", markeredgewidth=1,
                    markeredgecolor=color, linewidth=0.8, capsize=2,
                    capthick=0.5, zorder=5)

    p_str = "p < .001" if p_ic2 < 0.001 else f"p = {p_ic2:.3f}"
    ax.text(0.97, 0.97,
            r"$\beta_{IC^2}$ = " + f"{beta_ic2:.2f}\n{p_str}\nN = {len(sub)}",
            transform=ax.transAxes, fontsize=6, va="top", ha="right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_xlabel("Integrative complexity (z-scored)")
    ax.set_ylabel("Belief change (0–100)")
    ax.set_title(f"Study {s}", fontsize=8, fontweight="bold")

fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "si_figure2_three_study_replication.pdf", bbox_inches="tight")
fig.savefig(OUT / "si_figure2_three_study_replication.png", bbox_inches="tight")
print(f"Wrote {OUT}/si_figure2_three_study_replication.[pdf,png]")
