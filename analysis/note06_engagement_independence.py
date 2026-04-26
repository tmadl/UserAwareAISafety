#!/usr/bin/env python3
"""note06_engagement_independence.py — SI Note 6.

Reproduces SI Table tab:costello_engagement: the IC × engagement-moderator
hierarchy. For each engagement proxy M (topic importance, pre-belief
strength, word count), fits the canonical Costello quadratic with the full
IC×M + IC²×M interaction hierarchy and reports β/p/BF for each interaction
plus the joint BF against the no-interaction alternative.

Result claim (SI Note 6): "Every first-order Bayes factor is ≤ 0.04, every
second-order Bayes factor is ≤ 0.15, and every joint Bayes factor against
the two-interaction alternative is ≤ 0.005."

Reads:  data/costello2024/analysis_data.csv
        data/costello2024/texts_for_scoring.jsonl
        data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def load():
    an = pd.read_csv(DATA / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic_q = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC_q400": q["ic_qwenorpo400_logit"].astype(float).values,
    })
    df = an.merge(ic_q, on="participantId", how="left")
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "Importance", "IC_q400"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Costello self-report items use -999 as missing sentinel
    df.loc[df["Importance"] <= -90, "Importance"] = np.nan
    return df


def fit_hierarchy(df, mod_col, ctrl_cols):
    """For moderator M, fit:
      m_red:  DV ~ IC + IC² + M + ctrl                 (no interactions)
      m_li:   m_red + IC×M                             (linear interaction added)
      m_full: m_li + IC²×M                             (quadratic interaction added)

    Returns β, p for IC×M and IC²×M from m_full; BF₁₀ for each over its
    nested predecessor; joint BF (m_full vs m_red).
    """
    need = ["DV_BeliefChange_Specific", "IC_q400", mod_col] + ctrl_cols
    sub = df.dropna(subset=need).copy()
    y = sub["DV_BeliefChange_Specific"].values.astype(float)
    ic = zs(sub["IC_q400"].values)
    ic2 = zs(sub["IC_q400"].values ** 2)
    m = zs(sub[mod_col].values)
    icm = ic * m
    ic2m = ic2 * m
    ctrls = [zs(sub[c].values) for c in ctrl_cols]

    X_red = sm.add_constant(np.column_stack([ic, ic2, m] + ctrls))
    X_li = sm.add_constant(np.column_stack([ic, ic2, m, icm] + ctrls))
    X_full = sm.add_constant(np.column_stack([ic, ic2, m, icm, ic2m] + ctrls))

    m_red = sm.OLS(y, X_red).fit()
    m_li = sm.OLS(y, X_li).fit()
    m_full = sm.OLS(y, X_full).fit()

    bf_icm = float(np.exp((m_red.bic - m_li.bic) / 2))      # IC×M alone (over no-interaction)
    bf_ic2m = float(np.exp((m_li.bic - m_full.bic) / 2))    # IC²×M added on top of IC×M
    bf_joint = float(np.exp((m_red.bic - m_full.bic) / 2))  # both vs neither

    # Coefficient positions in m_full: const(0), IC(1), IC²(2), M(3), IC×M(4), IC²×M(5), ctrls(...)
    return dict(
        n=len(sub),
        b_icm=m_full.params[4], p_icm=m_full.pvalues[4], bf_icm=bf_icm,
        b_ic2m=m_full.params[5], p_ic2m=m_full.pvalues[5], bf_ic2m=bf_ic2m,
        bf_joint=bf_joint,
    )


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def main():
    df = load()
    print(f"Loaded {len(df)} Costello participants\n")

    print("SI Note 6 — Costello engagement-independence")
    print("Table: tab:costello_engagement\n")

    mods = [
        ("Importance",             "Topic importance",
         ["Pre_Belief_Specific", "OpenendedResponseWordCount"]),
        ("Pre_Belief_Specific",    "Pre-belief strength",
         ["OpenendedResponseWordCount"]),
        ("OpenendedResponseWordCount", "Word count",
         ["Pre_Belief_Specific"]),
    ]
    print(f"{'Moderator M':<22s} {'N':>5s}  "
          f"{'β(IC×M)':>9s} {'p':>7s} {'BF':>6s}  "
          f"{'β(IC²×M)':>10s} {'p':>7s} {'BF':>6s}  "
          f"{'BF(joint)':>10s}")
    print("-" * 95)

    for mod_col, mod_label, ctrls in mods:
        r = fit_hierarchy(df, mod_col, ctrls)
        print(f"{mod_label:<22s} {r['n']:>5d}  "
              f"{r['b_icm']:>+9.2f} {fmt_p(r['p_icm']):>7s} {r['bf_icm']:>6.3f}  "
              f"{r['b_ic2m']:>+10.2f} {fmt_p(r['p_ic2m']):>7s} {r['bf_ic2m']:>6.3f}  "
              f"{r['bf_joint']:>10.4f}")

    print("\nClaim check (SI Note 6):")
    print("  All first-order BFs ≤ 0.04   — IC×M  Bayes factors above")
    print("  All second-order BFs ≤ 0.15  — IC²×M Bayes factors above")
    print("  All joint BFs ≤ 0.005        — joint Bayes factors above")


if __name__ == "__main__":
    main()
