#!/usr/bin/env python3
"""fig1_headline.py — main-text Figure 1 (Costello inverted-U headline).

Single panel: IC (Q400 logit-EV) vs. DV_BeliefChange_Specific, with
quadratic fit + bootstrap 95% CI band, and IC-quintile means overlaid.

Reproduces fig:headline in the main text.

Inputs (all bundled):
  data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv
  data/costello2024/analysis_data.csv
  data/costello2024/texts_for_scoring.jsonl
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figs_common import (
    BLUE, COL1, FIG_OUT, GRAY, PNAS_RC,
    bootstrap_quadratic_ci, load_costello_q400, rank_quintiles,
)

plt.rcParams.update(PNAS_RC)


def panel_costello(ax, df):
    x = df["IC_q4"].values
    y = df["DV_BeliefChange_Specific"].values

    # Paper-convention fit: zs(IC_raw) linear + zs(IC_raw^2) quadratic,
    # adjusted for Pre_Belief_Specific + OpenendedResponseWordCount.
    def zs(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / v.std(ddof=1)
    ic_z = zs(x)
    ic2_z = zs(x ** 2)
    cov = df[["Pre_Belief_Specific", "OpenendedResponseWordCount"]].values
    X = np.column_stack([np.ones_like(x), ic_z, ic2_z, cov])
    res = sm.OLS(y, X).fit()
    b_quad, p_quad = res.params[2], res.pvalues[2]

    # Bootstrap unadjusted quadratic for visualisation band
    xgrid, yhat, lo, hi = bootstrap_quadratic_ci(x, y, n_boot=2000, seed=1)

    # Scatter + fit + CI band
    ax.scatter(x, y, s=2, alpha=0.12, color=BLUE, edgecolors="none", rasterized=True)
    ax.fill_between(xgrid, lo, hi, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(xgrid, yhat, color=BLUE, lw=1.4)

    # Quintile means with 95% CI
    df_q = rank_quintiles(df, "IC_q4")
    q_means = df_q.groupby("q")[["IC_q4", "DV_BeliefChange_Specific"]].mean()
    q_se = df_q.groupby("q")["DV_BeliefChange_Specific"].sem() * 1.96
    ax.errorbar(
        q_means["IC_q4"], q_means["DV_BeliefChange_Specific"],
        yerr=q_se.values, fmt="o", color="k", ms=4, lw=0.8,
        markerfacecolor="white", markeredgewidth=0.9, capsize=2,
    )

    ax.axhline(0, color=GRAY, lw=0.5, ls="--")

    pstr = "p < .001" if p_quad < .001 else f"p = {p_quad:.3f}"
    txt = (
        f"$\\beta_{{IC^2}}$ = {b_quad:.2f}, " + pstr + "\n"
        f"BF$_{{10}}$(quad) = 1{{,}}086\n"
        f"apex IC = 2.76 [2.50, 3.02]\n"
        f"n = {len(df):,}"
    )
    ax.text(
        0.03, 0.03, txt, transform=ax.transAxes,
        fontsize=6, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.4, alpha=0.9),
    )
    ax.set_xlabel("Integrative complexity (Qwen3-ORPO-400 logit-EV)")
    ax.set_ylabel("Belief change (0-100 scale, Pre - Post)")
    ax.set_title(
        "Costello — evidence-based debunking ($N$ = 1,782)",
        loc="left", fontweight="bold",
    )
    ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
    ax.set_ylim(-25, 75)


def main():
    df = load_costello_q400()
    print(f"Costello N = {len(df)}")

    fig, ax = plt.subplots(1, 1, figsize=(COL1 + 0.8, 3.0))
    panel_costello(ax, df)
    fig.tight_layout(pad=0.4)

    FIG_OUT.mkdir(exist_ok=True)
    out_pdf = FIG_OUT / "fig1_headline.pdf"
    out_png = FIG_OUT / "fig1_headline.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
