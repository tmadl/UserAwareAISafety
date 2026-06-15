#!/usr/bin/env python3
"""note21_reception_demand.py — SI Note 21.

Reception-demand modulation of the user-IC inverted-U: median-split the
sample on AI-side argument density (evidence-cue words, numeric
references, proper-noun density across the three GPT debunking turns) and
refit the paper-spec quadratic moderation within each half. McGuire's
reception--yielding model predicts the apex shifts rightward (toward
higher IC) under more cognitively demanding AI counter-arguments.

Reproduces SI Table tab:reception_demand:
  Low-demand half:  beta_IC^2 ~ -8,    BF10 ~ 0.16, apex ~ +2.42
  High-demand half: beta_IC^2 ~ -23,   BF10 ~ 8453, apex ~ +2.81
  Apex shift ~ +0.38 IC units

Plus the reverse-causal check: residualising the demand composite on
user IC before the median split leaves the result essentially unchanged.

Pre-computed bundled CSV: data/costello2024/costello_demand_composite.csv
(deterministic from raw GPT response text — only the per-participant
density features and composite are bundled, no raw text).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"


def zs(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def fit_quadratic(df):
    """Paper-spec quadratic within a sample half.

    Reports both curvature parameterisations: b_q is on the z(IC^2) scale
    (zscore of IC^2; the larger-magnitude legacy scale) and b_q_zic2 is on
    the headline z(IC)^2 scale (zscore of IC, then squared) used in the SI
    Table tab:reception_demand and the main text. The BF and the raw-scale
    apex are identical across the two parameterisations (same fitted curve)."""
    s = df.dropna(subset=["IC", "DV_BeliefChange_Specific",
                          "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()
    y = s["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    raw = s["IC"].to_numpy(dtype=float)
    ic = zs(raw)
    ic2_legacy = zs(raw ** 2)        # z(IC^2) scale
    ic2_head = ic ** 2               # z(IC)^2 headline scale
    cov = [zs(s[c].values) for c in ("Pre_Belief_Specific", "OpenendedResponseWordCount")]
    X1 = sm.add_constant(np.column_stack([ic, *cov]))
    X2 = sm.add_constant(np.column_stack([ic, ic2_legacy, *cov]))
    X2h = sm.add_constant(np.column_stack([ic, ic2_head, *cov]))
    m1 = sm.OLS(y, X1).fit(); m2 = sm.OLS(y, X2).fit(); m2h = sm.OLS(y, X2h).fit()
    bf = float(np.exp((m1.bic - m2.bic) / 2))
    sig1, sig2 = float(np.std(raw, ddof=0)), float(np.std(raw ** 2, ddof=0))
    b_lin, b_q = float(m2.params[1]), float(m2.params[2])
    apex = -b_lin * sig2 / (2 * b_q * sig1) if b_q != 0 else np.nan
    # headline-scale (z(IC)^2) coefficients + matching apex on the raw scale
    bl_h, bq_h = float(m2h.params[1]), float(m2h.params[2])
    apex_h = raw.mean() - bl_h / (2 * bq_h) * sig1 if bq_h != 0 else np.nan
    return dict(n=len(s), b_lin=b_lin, b_q=b_q, p_q=float(m2.pvalues[2]),
                BF10=bf, apex=apex,
                b_lin_zic2=bl_h, b_q_zic2=bq_h, apex_zic2=apex_h)


def report_split(df, split_col, label):
    med = df[split_col].median()
    lo = df[df[split_col] <= med]
    hi = df[df[split_col] >  med]
    r_lo, r_hi = fit_quadratic(lo), fit_quadratic(hi)
    print(f"\n  {label} (median = {med:+.3f}):")
    print(f"    {'Half':<14}{'n':>5}  {'beta_lin':>9}  {'beta_quad':>10}  "
          f"{'p_quad':>7}  {'BF10':>9}  {'apex':>6}")
    for name, r in (("low demand ", r_lo), ("high demand", r_hi)):
        print(f"    {name:<14}{r['n']:>5}  {r['b_lin']:>+9.3f}  {r['b_q']:>+10.3f}  "
              f"{r['p_q']:>7.3f}  {r['BF10']:>9.2f}  {r['apex']:>+6.2f}")
    print(f"    delta(apex) = {r_hi['apex'] - r_lo['apex']:+.2f} IC units")
    return r_lo, r_hi


def main():
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    ic = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ad = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)
    demand = pd.read_csv(DATA / "costello_demand_composite.csv")

    base = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC":             ic["ic_qwenorpo400_logit"].astype(float).values,
    })
    df = (base.merge(ad[["participantId", "DV_BeliefChange_Specific",
                         "Pre_Belief_Specific", "OpenendedResponseWordCount"]],
                     on="participantId", how="inner")
              .merge(demand, on="participantId", how="inner"))
    df = df.dropna(subset=["IC", "DV_BeliefChange_Specific",
                            "Pre_Belief_Specific", "OpenendedResponseWordCount",
                            "demand_composite"]).reset_index(drop=True)

    print(f"SI Note 21 — Reception-demand modulation of the inverted-U")
    print(f"Complete-cases N = {len(df)}")

    ref = fit_quadratic(df)
    print(f"\nFull-sample reference (paper spec):")
    print(f"  n={ref['n']}  beta_IC^2 = {ref['b_q']:+.3f}  "
          f"p = {ref['p_q']:.4f}  BF10 = {ref['BF10']:.1f}  apex = {ref['apex']:+.2f}")

    r_lo, r_hi = report_split(df, "demand_composite", "Median split on AI demand composite")

    # SI Table tab:reception_demand is on the headline z(IC)^2 scale.
    print("\n  -> SI Table tab:reception_demand (median-AI-demand split, z(IC)^2 scale):")
    for name, r in (("Low demand ", r_lo), ("High demand", r_hi)):
        print(f"     {name:<12} n={r['n']:>4}  beta_lin={r['b_lin_zic2']:+.3f}  "
              f"beta_IC^2={r['b_q_zic2']:+.3f}  BF10={r['BF10']:.2f}  apex={r['apex_zic2']:+.3f}")
    print(f"     delta(apex) = {r_hi['apex_zic2'] - r_lo['apex_zic2']:+.3f} IC units")

    # Per-component apex shifts (median split on each density component).
    print("\n  Per-component apex shifts (median split on each density component, z(IC)^2 apex):")
    for col, lab in (("evid_100w_mean", "evidence-cue words"),
                     ("num_100w_mean", "numeric references"),
                     ("prop_100w_mean", "proper-noun density")):
        med_c = df[col].median()
        a_lo = fit_quadratic(df[df[col] <= med_c])["apex_zic2"]
        a_hi = fit_quadratic(df[df[col] >  med_c])["apex_zic2"]
        print(f"     {lab:<22} apex_lo={a_lo:+.3f}  apex_hi={a_hi:+.3f}  shift={a_hi - a_lo:+.3f}")

    # Reverse-causal: residualise demand on user IC, re-split
    s = df.dropna(subset=["IC", "demand_composite"]).copy()
    X = sm.add_constant(zs(s["IC"].values))
    m = sm.OLS(s["demand_composite"].values, X).fit()
    s["demand_resid"] = s["demand_composite"].values - m.fittedvalues
    print(f"\nReverse-causal check: user IC -> AI demand composite")
    print(f"  beta = {m.params[1]:+.4f}  p = {m.pvalues[1]:.4f}  R^2 = {m.rsquared:.4f}")
    report_split(s, "demand_resid", "Median split on residualised demand (AI demand | user IC)")


if __name__ == "__main__":
    main()
