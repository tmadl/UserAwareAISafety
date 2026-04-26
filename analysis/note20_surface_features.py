#!/usr/bin/env python3
"""note20_surface_features.py — SI Note 20.

Surface-feature ablation: tests whether the IC inverted-U is a re-labelling
of more familiar surface text properties (verbosity, readability, vocabulary
diversity, discourse-marker density). For each surface feature we fit the
paper-spec quadratic model standalone, then check whether the IC^2 Bayes
factor survives when all 7 surface features (linear + quadratic = 14 terms)
are included as covariates.

Surface features (deterministic from pre-treatment text):
  surf_wc      word count
  surf_fk      Flesch-Kincaid grade
  surf_dc      Dale-Chall readability
  surf_smog    SMOG index
  surf_asl     average sentence length (words per sentence)
  surf_ttr     type-token ratio
  surf_marker  discourse-marker density per 100 words

Pre-computed bundled CSV: data/costello2024/costello_surface_features.csv
(deterministic from the bundled `texts_for_scoring.jsonl`; no raw download
required for this script).

Reproduces SI Table tab:surface_ablation:
  - Standalone quadratic BFs <= 1.4 for every surface feature
  - Omnibus IC^2 BF attenuates from ~1086 to ~47 with all 14 surface
    terms as covariates; beta_IC^2 retains a strongly significant
    negative quadratic.
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


def quadratic_fit(df, pred, covars):
    y = df["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    raw = df[pred].to_numpy(dtype=float)
    lin, quad = zs(raw), zs(raw ** 2)
    cov_z = [zs(df[c].to_numpy(dtype=float)) for c in covars]
    X1 = sm.add_constant(np.column_stack([lin, *cov_z]))
    X2 = sm.add_constant(np.column_stack([lin, quad, *cov_z]))
    m1 = sm.OLS(y, X1).fit()
    m2 = sm.OLS(y, X2).fit()
    bf = float(np.exp((m1.bic - m2.bic) / 2))
    return dict(b_lin=float(m2.params[1]), p_lin=float(m2.pvalues[1]),
                b_q=float(m2.params[2]), p_q=float(m2.pvalues[2]),
                BF10=bf, r2_lin=float(m1.rsquared), r2_quad=float(m2.rsquared))


def main():
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    ic = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ad = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)
    surf = pd.read_csv(DATA / "costello_surface_features.csv")

    base = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC_q4":          ic["ic_qwenorpo400_logit"].astype(float).values,
    })
    df = (base.merge(surf, on="participantId", how="inner")
              .merge(ad[["participantId", "DV_BeliefChange_Specific",
                         "Pre_Belief_Specific", "OpenendedResponseWordCount"]],
                     on="participantId", how="inner"))

    surf_cols = ["surf_wc", "surf_fk", "surf_dc", "surf_smog",
                 "surf_asl", "surf_ttr", "surf_marker"]
    df = df.dropna(subset=["DV_BeliefChange_Specific", "Pre_Belief_Specific",
                            "IC_q4", *surf_cols]).reset_index(drop=True)

    print("SI Note 20 — Surface-feature ablation of the user-IC curvature")
    print(f"Complete-cases N = {len(df)}\n")

    COV = ("Pre_Belief_Specific", "OpenendedResponseWordCount")

    print("Standalone quadratic moderation on each surface feature (paper spec):")
    print(f"  {'Predictor':<14}  {'beta_lin':>9}  {'beta_quad':>9}  {'p_quad':>7}  {'BF10':>8}")
    for p in surf_cols + ["IC_q4"]:
        r = quadratic_fit(df, p, COV)
        print(f"  {p:<14}  {r['b_lin']:>+9.3f}  {r['b_q']:>+9.3f}  "
              f"{r['p_q']:>7.3f}  {r['BF10']:>8.2f}")

    # Omnibus ablation
    y = df["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    ic_lin = zs(df["IC_q4"].values); ic_q = zs(df["IC_q4"].values ** 2)
    base_cov = [zs(df[c].values) for c in COV]
    surf_lin = [zs(df[c].values) for c in surf_cols]
    surf_q   = [zs(df[c].values ** 2) for c in surf_cols]

    X_no_ic  = sm.add_constant(np.column_stack([*base_cov, *surf_lin, *surf_q]))
    X_ic_lin = sm.add_constant(np.column_stack([ic_lin, *base_cov, *surf_lin, *surf_q]))
    X_ic_q   = sm.add_constant(np.column_stack([ic_lin, ic_q, *base_cov, *surf_lin, *surf_q]))
    m0 = sm.OLS(y, X_no_ic).fit()
    m_lin = sm.OLS(y, X_ic_lin).fit()
    m_q = sm.OLS(y, X_ic_q).fit()
    bf_q_vs_lin = float(np.exp((m_lin.bic - m_q.bic) / 2))

    print(f"\nOmnibus ablation (IC + IC^2 added to a model with all 7 surface lin+quad terms):")
    print(f"  Baseline (covars + 14 surface terms, no IC) R^2 = {m0.rsquared:.4f}")
    print(f"  + IC (linear)          R^2 = {m_lin.rsquared:.4f}  delta R^2 = {m_lin.rsquared - m0.rsquared:+.4f}")
    print(f"  + IC^2 (quadratic)     R^2 = {m_q.rsquared:.4f}  delta R^2 = {m_q.rsquared - m_lin.rsquared:+.4f}")
    print(f"  beta_IC^2 (adjusted) = {m_q.params[2]:+.3f}  p = {m_q.pvalues[2]:.4f}")
    print(f"  BF10 (quadratic vs linear, with surface controls) = {bf_q_vs_lin:.2f}")

    r_ref = quadratic_fit(df, "IC_q4", COV)
    print(f"\nReference (IC alone, no surface controls): BF10 = {r_ref['BF10']:.1f}")
    print(f"After omnibus surface controls:             BF10 = {bf_q_vs_lin:.2f}")


if __name__ == "__main__":
    main()
