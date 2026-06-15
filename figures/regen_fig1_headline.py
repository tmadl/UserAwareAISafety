"""Regenerate figures_v2/fig1_headline with canonical z(IC)^2 parameterisation.

Identical to scripts/figures/fig1_headline.py except:
  - ic2_z = ic_z ** 2  (z(IC) squared) instead of zs(x ** 2)  (z of IC^2)
  - output path -> paper/PNAS_Nexus_v3/figures_v2
"""
import sys
from pathlib import Path
sys.path.insert(0, "/mnt/workvm/UserAwareAISafety/scripts/figures")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (
    PNAS_RC, BLUE, GRAY, COL1,
    load_costello_q400,
    bootstrap_quadratic_ci,
    rank_quintiles,
)

plt.rcParams.update(PNAS_RC)
FIG_OUT = Path("/mnt/workvm/UserAwareAISafety/paper/PNAS_Nexus_v3/figures_v2")


def panel_costello(ax, df):
    import statsmodels.api as sm
    x = df["IC_q4"].values
    y = df["DV_BeliefChange_Specific"].values

    # Paper-canonical fit: z(IC) linear + z(IC)^2 quadratic,
    # adjusted for z(pre-belief) + z(word count).
    def zs(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / v.std(ddof=1)
    ic_z = zs(x)
    ic2_z = ic_z ** 2
    cov = np.column_stack([zs(df["Pre_Belief_Specific"].values),
                           zs(df["OpenendedResponseWordCount"].values)])
    X = np.column_stack([np.ones_like(x), ic_z, ic2_z, cov])
    res = sm.OLS(y, X).fit()
    q_fit = {"b_quad": res.params[2], "p_quad": res.pvalues[2]}
    print(f"beta_IC2 = {q_fit['b_quad']:.3f}, p = {q_fit['p_quad']:.2e}")

    xgrid, yhat, lo, hi = bootstrap_quadratic_ci(x, y, n_boot=2000, seed=1)

    # Flag the sparse high-IC region (>99th percentile) where the quadratic
    # extrapolates beyond meaningful data support: render the fit solid within
    # support and faded/dashed beyond, and grey-shade the sparse zone, so the
    # below-zero extrapolation cannot be misread as an estimated backfire.
    p99 = np.percentile(x, 99)
    insup = xgrid <= p99
    ax.scatter(x, y, s=2, alpha=0.12, color=BLUE, edgecolors="none", rasterized=True)
    ax.fill_between(xgrid[insup], lo[insup], hi[insup], color=BLUE, alpha=0.18,
                    linewidth=0)
    ax.plot(xgrid[insup], yhat[insup], color=BLUE, lw=1.4)
    ax.plot(xgrid[~insup], yhat[~insup], color=BLUE, lw=1.0, ls=(0, (3, 2)),
            alpha=0.45)
    ax.axvspan(p99, xgrid.max() + 0.2, color=GRAY, alpha=0.10, linewidth=0)
    ax.text(p99 + 0.05, 70, "sparse\n(<1% of sample)", fontsize=5,
            color=GRAY, va="top", ha="left")

    df_q = rank_quintiles(df, "IC_q4")
    q_means = df_q.groupby("q")[["IC_q4", "DV_BeliefChange_Specific"]].mean()
    q_se = df_q.groupby("q")["DV_BeliefChange_Specific"].sem() * 1.96
    ax.errorbar(q_means["IC_q4"], q_means["DV_BeliefChange_Specific"],
                yerr=q_se.values, fmt="o", color="k", ms=4, lw=0.8,
                markerfacecolor="white", markeredgewidth=0.9, capsize=2)

    ax.axhline(0, color=GRAY, lw=0.5, ls="--")

    pstr = "p < .001" if q_fit["p_quad"] < .001 else f"p = {q_fit['p_quad']:.3f}"
    txt = (f"β$_{{IC^2}}$ = {q_fit['b_quad']:.2f}, " + pstr + "\n"
           f"BF$_{{10}}$(quad) = 1,086\n"
           f"apex IC = 2.75 [2.55, 2.99]\n"
           f"n = {len(df):,}")
    ax.text(0.03, 0.03, txt, transform=ax.transAxes,
            fontsize=6, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=GRAY, lw=0.4, alpha=0.9))

    ax.set_xlabel("Integrative complexity (primary scorer, logit-EV)")
    ax.set_ylabel("Belief change (0–100 scale, Pre − Post)")
    ax.set_title("Costello — evidence-based debunking ($N$ = 1,782)",
                 loc="left", fontweight="bold")
    ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
    ax.set_ylim(-25, 75)


def main():
    co = load_costello_q400()
    print(f"Costello N = {len(co)}")
    fig, ax = plt.subplots(1, 1, figsize=(COL1 + 0.8, 3.0))
    panel_costello(ax, co)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG_OUT / "fig1_headline.pdf")
    fig.savefig(FIG_OUT / "fig1_headline.png")
    print(f"Wrote {FIG_OUT}/fig1_headline.pdf and .png")


if __name__ == "__main__":
    main()
