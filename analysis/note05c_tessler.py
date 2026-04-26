#!/usr/bin/env python3
"""note05c_tessler.py — SI Note 5c (Tessler/Habermas).

Tessler et al. (2024) democratic deliberation corpus, N = 6,024 participants
across 17,919 participant x question observations. AI mediator helps groups
find common ground; belief change is measured but the intended outcome is
consensus, not directional persuasion.

Tests whether IC moderates belief change in this paradigm. The SI claim is
"evidence of absence" (BF for curvature <= 0.02) on both primary DVs, plus
null linear slopes; cluster-robust SEs on participantId do not change the
conclusion. This is one of the boundary-condition datasets that defines
where IC moderation is and is not operative (Note 5).

Inputs (all bundled):
  data/tessler2024/analysis_data.csv         (DVs + covariates)
  data/tessler2024/tessler_ic_qwenorpo400.csv (Q400 logit-EV IC scores)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "tessler2024"


def zs(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def fit(df, dv, cluster=False):
    s = df.dropna(subset=[dv, "ic_ev", "wc_initial"]).copy()
    y = s[dv].to_numpy(dtype=float)
    raw = s["ic_ev"].to_numpy(dtype=float)
    ic = zs(raw); ic2 = zs(raw ** 2)
    wc = zs(s["wc_initial"].to_numpy(dtype=float))
    X1 = sm.add_constant(np.column_stack([ic, wc]))
    X2 = sm.add_constant(np.column_stack([ic, ic2, wc]))
    if cluster:
        groups = s["metadata_participant_id"].astype(str).values
        m1 = sm.OLS(y, X1).fit(cov_type="cluster", cov_kwds={"groups": groups})
        m2 = sm.OLS(y, X2).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        m1 = sm.OLS(y, X1).fit()
        m2 = sm.OLS(y, X2).fit()
    bf = float(np.exp((m1.bic - m2.bic) / 2))
    return dict(n=len(s),
                b_lin=float(m2.params[1]), p_lin=float(m2.pvalues[1]),
                b_q=float(m2.params[2]), p_q=float(m2.pvalues[2]),
                BF10=bf)


def main():
    ad = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)
    ic = pd.read_csv(DATA / "tessler_ic_qwenorpo400.csv")
    df = ad.merge(ic[["participantId", "ic_ev"]], on="participantId", how="inner")
    for c in ["belief_change", "abs_belief_change", "wc_initial", "ic_ev"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"SI Note 5c — Tessler/Habermas deliberation corpus")
    print(f"observations: {len(df)}  unique participants: {df['metadata_participant_id'].nunique()}\n")

    print(f"{'DV':<20}{'n':>6}  {'beta_lin':>10}  {'p_lin':>7}  "
          f"{'beta_quad':>10}  {'p_quad':>7}  {'BF10':>8}")
    for dv in ["belief_change", "abs_belief_change"]:
        r = fit(df, dv)
        print(f"  {dv:<18}{r['n']:>6}  {r['b_lin']:>+10.4f}  {r['p_lin']:>7.3f}  "
              f"{r['b_q']:>+10.4f}  {r['p_q']:>7.3f}  {r['BF10']:>8.3f}")

    print(f"\nCluster-robust SEs on metadata_participant_id "
          f"(robustness to within-person clustering):")
    print(f"{'DV':<20}{'n':>6}  {'beta_lin':>10}  {'p_lin':>7}  "
          f"{'beta_quad':>10}  {'p_quad':>7}  {'BF10':>8}")
    for dv in ["belief_change", "abs_belief_change"]:
        r = fit(df, dv, cluster=True)
        print(f"  {dv:<18}{r['n']:>6}  {r['b_lin']:>+10.4f}  {r['p_lin']:>7.3f}  "
              f"{r['b_q']:>+10.4f}  {r['p_q']:>7.3f}  {r['BF10']:>8.3f}")

    print("\nSI claim: pooled BF for curvature <= 0.02 on both primary DVs;")
    print("linear slopes null; cluster-robust SEs do not change the conclusion.")


if __name__ == "__main__":
    main()
