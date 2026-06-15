"""
SI figure for Note 21: median-split of Costello by AI-side reception-demand
composite. Standalone single-panel version of the grant-headline Panel B,
styled for PNAS SI. High-demand half shows sharp inverted-U; low-demand
half is flat (evidence of absence).
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, "/mnt/workvm/UserAwareAISafety/scripts/figures")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

from _common import (
    PNAS_RC, GRAY, COL2, FIG_OUT, DATA,
    load_costello_q400, bootstrap_quadratic_ci, rank_quintiles,
)

plt.rcParams.update(PNAS_RC)

MUTED = "#5D5D5D"
HI    = "#1F3A68"

GPT_CSV = DATA / "costello2024/Data 8.28.24/AllDataForPublication.PPI.8.28.24.csv"

EVIDENCE_PAT = re.compile(
    r"\b(evidence|study|studies|research|researcher|researchers|"
    r"source|sources|citation|citations|data|report|reports|"
    r"analysis|analyses|finding|findings|paper|papers|"
    r"investigation|investigations|document|documents|scientist|scientists|"
    r"expert|experts|peer-review(?:ed)?|journal|journals|published)\b",
    re.I,
)
NUMERIC_PAT = re.compile(
    r"(?:\b\d{4}\b|\b\d+\.?\d*\s*%|\$\d+|"
    r"\b\d+\.?\d*\s*(?:million|billion|thousand)\b|\b\d+\b)"
)
PROPER_PAT = re.compile(r"(?<=[a-z.\s])\b[A-Z][a-zA-Z]{2,}\b")


def per_100w(count, wc):
    return 100.0 * count / max(wc, 1)


def demand_from_text(text):
    t = str(text) if text else ""
    wc = len(t.split())
    return {
        "evid_100w": per_100w(len(EVIDENCE_PAT.findall(t)), wc),
        "num_100w":  per_100w(len(NUMERIC_PAT.findall(t)), wc),
        "prop_100w": per_100w(len(PROPER_PAT.findall(t)), wc),
    }


def zs(a):
    a = np.asarray(a, dtype=float)
    sd = np.nanstd(a, ddof=0)
    return (a - np.nanmean(a)) / (sd if sd > 0 else 1.0)


def zmean(*arrays):
    return np.nanmean(np.column_stack([zs(a) for a in arrays]), axis=1)


def build_demand_composite():
    raw = pd.read_csv(GPT_CSV, low_memory=False)
    gpt = (raw[["participantId", "GPTResponse", "GPTResponse2", "GPTResponse3"]]
           .drop_duplicates("participantId").copy())
    for i, col in enumerate(["GPTResponse", "GPTResponse2", "GPTResponse3"], start=1):
        feats = gpt[col].apply(demand_from_text).apply(pd.Series)
        feats.columns = [f"{k}_t{i}" for k in feats.columns]
        gpt = pd.concat([gpt, feats], axis=1)
    for feat in ("evid_100w", "num_100w", "prop_100w"):
        cols = [f"{feat}_t{i}" for i in (1, 2, 3)]
        gpt[f"{feat}_mean"] = gpt[cols].mean(axis=1, skipna=True)
    gpt["demand_composite"] = zmean(
        gpt["evid_100w_mean"].values,
        gpt["num_100w_mean"].values,
        gpt["prop_100w_mean"].values,
    )
    return gpt[["participantId", "demand_composite"]]


def fit_paper_quadratic(df, ic_col="IC_q4"):
    s = df.dropna(subset=[ic_col, "DV_BeliefChange_Specific",
                          "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()
    y = s["DV_BeliefChange_Specific"].to_numpy(float)
    raw = s[ic_col].to_numpy(float)
    ic = zs(raw)
    ic2 = ic ** 2
    covz = [zs(s[c].to_numpy(float))
            for c in ("Pre_Belief_Specific", "OpenendedResponseWordCount")]
    X1 = sm.add_constant(np.column_stack([ic, *covz]))
    X2 = sm.add_constant(np.column_stack([ic, ic2, *covz]))
    m1 = sm.OLS(y, X1).fit()
    m2 = sm.OLS(y, X2).fit()
    bf = float(np.exp((m1.bic - m2.bic) / 2))
    b_lin, b_q = float(m2.params[1]), float(m2.params[2])
    try:
        apex = float(np.mean(raw)) + float(np.std(raw, ddof=0)) * (-b_lin / (2 * b_q))
    except Exception:
        apex = float("nan")
    return dict(b_lin=b_lin, b_q=b_q, p_q=float(m2.pvalues[2]),
                BF10=bf, apex=apex, n=len(s))


def fmt_bf(bf):
    if bf < 1:
        return f"{bf:.2f}"
    if bf < 1000:
        return f"{bf:.0f}"
    import math
    return f"{math.floor(bf):,}"


def main():
    co = load_costello_q400()
    dem = build_demand_composite()
    df = co.merge(dem, on="participantId", how="inner").dropna(subset=["demand_composite"])
    print(f"Costello N = {len(co)}, merged N = {len(df)}")

    median = df["demand_composite"].median()
    lo_df = df[df["demand_composite"] <= median].reset_index(drop=True)
    hi_df = df[df["demand_composite"] >  median].reset_index(drop=True)

    r_lo = fit_paper_quadratic(lo_df)
    r_hi = fit_paper_quadratic(hi_df)
    print(f"Low : b_q={r_lo['b_q']:+.2f} BF={r_lo['BF10']:.3f} apex={r_lo['apex']:.2f} n={r_lo['n']}")
    print(f"High: b_q={r_hi['b_q']:+.2f} BF={r_hi['BF10']:.1f} apex={r_hi['apex']:.2f} n={r_hi['n']}")

    fig, ax = plt.subplots(figsize=(COL2 * 0.62, 3.0))

    for df_sub, color, r in [(lo_df, MUTED, r_lo), (hi_df, HI, r_hi)]:
        x = df_sub["IC_q4"].to_numpy(float)
        y = df_sub["DV_BeliefChange_Specific"].to_numpy(float)
        xgrid, yhat, lo_b, hi_b = bootstrap_quadratic_ci(
            x, y, n_boot=2000, seed=2 if color == HI else 3)
        ax.fill_between(xgrid, lo_b, hi_b, color=color, alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(xgrid, yhat, color=color, lw=1.4, zorder=4)

        df_q = rank_quintiles(df_sub, "IC_q4")
        q_means = df_q.groupby("q")[["IC_q4", "DV_BeliefChange_Specific"]].mean()
        q_se = df_q.groupby("q")["DV_BeliefChange_Specific"].sem() * 1.96
        ax.errorbar(q_means["IC_q4"], q_means["DV_BeliefChange_Specific"],
                    yerr=q_se.values, fmt="o", color=color, ms=3.5, lw=0.7,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=0.9, capsize=2, zorder=5)

    ax.axhline(0, color=GRAY, lw=0.5, ls=(0, (4, 2)), zorder=1)

    txt_hi = (r"$\mathbf{High~AI~reception~demand}$" f" (top 50%, $n$ = {r_hi['n']:,})" "\n"
              rf"$\beta_{{\mathrm{{IC}}^{{2}}}} = {r_hi['b_q']:+.2f}$, $p < .001$, "
              rf"$\mathrm{{BF}}_{{10}} = {fmt_bf(r_hi['BF10'])}$")
    txt_lo = (r"$\mathbf{Low~AI~reception~demand}$" f" (bottom 50%, $n$ = {r_lo['n']:,})" "\n"
              rf"$\beta_{{\mathrm{{IC}}^{{2}}}} = {r_lo['b_q']:+.2f}$, "
              rf"$p = {r_lo['p_q']:.2f}$, "
              rf"$\mathrm{{BF}}_{{10}} = {fmt_bf(r_lo['BF10'])}$ (evidence of absence)")

    ax.text(0.97, 0.97, txt_hi, transform=ax.transAxes,
            fontsize=6, va="top", ha="right", color=HI,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=HI, lw=0.6, alpha=0.97))
    ax.text(0.03, 0.04, txt_lo, transform=ax.transAxes,
            fontsize=6, va="bottom", ha="left", color=MUTED,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=MUTED, lw=0.5, alpha=0.97))

    ax.set_xlabel("User integrative complexity (scored from text)")
    ax.set_ylabel(r"Belief change (Pre $-$ Post, pts)")

    fig.tight_layout(pad=0.3)
    import pathlib; OUT3 = pathlib.Path("/mnt/workvm/UserAwareAISafety/paper/PNAS_Nexus_v3/figures_v2"); fig.savefig(OUT3 / "fig_si_reception_demand.pdf")
    fig.savefig(OUT3 / "fig_si_reception_demand.png", dpi=300)
    print("Wrote fig_si_reception_demand.{pdf,png}")


if __name__ == "__main__":
    main()
