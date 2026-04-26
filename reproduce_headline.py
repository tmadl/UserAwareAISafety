#!/usr/bin/env python3
"""reproduce_headline.py — 30-second sanity check on the main-text headline.

Prints the four headline numbers from the Costello inverted-U with the
primary Q400 logit-EV scorer:

  beta_IC^2 = -15.17    p < 10^-6    BF_10 = 1086    apex = 2.76 [2.50, 3.02]

If any number drifts beyond rounding, something has changed downstream
of the bundled scored data; investigate before trusting the rest.

For the full headline reproduction with bootstrap CI, within-study
replication, and the 24-moderator comparison, run
  analysis/01_costello_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=0)


def main():
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    pids = [m["participantId"] for m in meta]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic = pd.DataFrame({
        "participantId": pids,
        "IC": pd.to_numeric(q["ic_qwenorpo400_logit"], errors="coerce").values,
    })
    ana = pd.read_csv(DATA / "analysis_data.csv")
    df = (ana.merge(ic, on="participantId", how="inner")
             .dropna(subset=["IC", "DV_BeliefChange_Specific",
                              "Pre_Belief_Specific", "OpenendedResponseWordCount"]))
    n = len(df)
    y = df["DV_BeliefChange_Specific"].values.astype(float)
    raw = df["IC"].values.astype(float)
    ic_z, ic2_z = zs(raw), zs(raw ** 2)
    pre_z, wc_z = zs(df["Pre_Belief_Specific"].values), zs(df["OpenendedResponseWordCount"].values)

    X1 = sm.add_constant(np.column_stack([ic_z, pre_z, wc_z]))
    X2 = sm.add_constant(np.column_stack([ic_z, ic2_z, pre_z, wc_z]))
    m1 = sm.OLS(y, X1).fit()
    m2 = sm.OLS(y, X2).fit()
    bf = float(np.exp((m1.bic - m2.bic) / 2))
    b_lin, b_quad = float(m2.params[1]), float(m2.params[2])
    p_quad = float(m2.pvalues[2])

    sd_ic, sd_sq = float(np.std(raw, ddof=0)), float(np.std(raw ** 2, ddof=0))
    apex = -b_lin * sd_sq / (2 * b_quad * sd_ic)

    # Bootstrap apex CI
    rng = np.random.default_rng(42)
    peaks = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        rb = raw[idx]
        Xb = sm.add_constant(np.column_stack([
            zs(rb), zs(rb ** 2),
            zs(df["Pre_Belief_Specific"].values[idx]),
            zs(df["OpenendedResponseWordCount"].values[idx]),
        ]))
        try:
            mb = sm.OLS(y[idx], Xb).fit()
            sb_ic, sb_sq = np.std(rb, ddof=0), np.std(rb ** 2, ddof=0)
            peaks.append(-mb.params[1] * sb_sq / (2 * mb.params[2] * sb_ic))
        except Exception:
            pass
    lo, hi = np.percentile(peaks, [2.5, 97.5])

    print("Costello inverted-U headline (Q400 logit-EV, paper-spec model)")
    print(f"  n           = {n:,}")
    print(f"  beta_IC^2   = {b_quad:+.2f}   (paper: -15.17)")
    print(f"  p_IC^2      = {p_quad:.2e}   (paper: <1e-6)")
    print(f"  BF10(quad)  = {bf:,.1f}   (paper: 1,086)")
    print(f"  apex IC     = {apex:.2f} [{lo:.2f}, {hi:.2f}]   (paper: 2.76 [2.50, 3.02])")


if __name__ == "__main__":
    main()
