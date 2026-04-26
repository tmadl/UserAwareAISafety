#!/usr/bin/env python3
"""12_quintile_demographics.py — Per-IC-quintile demographic panel for SI Note 23.

Produces the panel that answers: who gets flagged by a bottom-IC-quintile screen?

  (a) Per-quintile demographic composition (education, age, gender, race,
      party, extremism).
  (b) Education-stratum reweighted screening retention: recompute the
      86.3% / 24.4% large-change / adverse-captured operating point after
      weighting quintiles to match the overall education distribution, to
      check whether the screen's value depends on the education-IC
      correlation.
  (c) Fairness sensitivity: test IC^2 x Education and IC^2 x Age
      interactions in the canonical quadratic moderation. Null or small
      interactions support the reading that the curvature is not just
      an education/age effect in disguise.

Reproduces the figures in SI Note 23 (tab:quintile_demographics,
tab:education_stratified_screen) from the canonical data sources.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data/costello2024/Data 8.28.24/AllDataForPublication.PPI.8.28.24.csv"
AD_CSV = ROOT / "data/costello2024/analysis_data.csv"
IC_CSV = ROOT / "data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv"
META_JSONL = ROOT / "data/costello2024/texts_for_scoring.jsonl"

ED_ORDER = {
    "LessThanHighSchool": 1, "HighSchool": 2, "SomeCollege": 3,
    "Associate": 4, "Bachelors": 5, "Masters": 6, "JD/MD": 7, "PhD": 8,
}


def zs(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def load_df():
    meta = [json.loads(l) for l in open(META_JSONL)]
    ic = pd.read_csv(IC_CSV)
    base = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC": ic["ic_qwenorpo400_logit"].astype(float).values,
    })
    ad = pd.read_csv(AD_CSV, low_memory=False)
    ad["Education_num"] = ad["Education_Cat"].map(ED_ORDER).astype(float)
    ad_cols = [
        "participantId", "DV_BeliefChange_Specific", "Pre_Belief_Specific",
        "OpenendedResponseWordCount", "Education_Cat", "Education_num",
        "AgeYears", "Extremism", "GenderCat", "PartyCat", "religion",
    ]
    from _raw_data_check import require_raw
    require_raw(RAW_CSV, "Costello", "https://osf.io/gdkb7/")
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    raw_cols = ["participantId", "Race_Cat"]
    raw = raw[raw_cols].drop_duplicates("participantId")
    df = base.merge(ad[ad_cols], on="participantId", how="inner")
    df = df.merge(raw, on="participantId", how="left")
    df = df.drop_duplicates("participantId").reset_index(drop=True)
    df = df.dropna(subset=["IC", "DV_BeliefChange_Specific",
                           "Pre_Belief_Specific",
                           "OpenendedResponseWordCount"]).reset_index(drop=True)
    df["IC_q"] = pd.qcut(df["IC"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return df


def pct(series, value):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return 100.0 * (s == value).mean()


def demographic_table(df):
    rows = []
    for q, g in df.groupby("IC_q", observed=True):
        rows.append({
            "Quintile": q,
            "n": len(g),
            "IC_mean": g["IC"].mean(),
            "Ed_mean": g["Education_num"].mean(),
            "Ed_BA+": 100.0 * (g["Education_num"] >= 5).mean(),
            "Age_mean": g["AgeYears"].mean(),
            "pct_Female": pct(g["GenderCat"], "Female"),
            "pct_Male": pct(g["GenderCat"], "Male"),
            "pct_White": pct(g["Race_Cat"], "White"),
            "pct_Black": pct(g["Race_Cat"], "Black"),
            "pct_Asian": pct(g["Race_Cat"], "Asian"),
            "pct_OtherRace": 100.0 - (pct(g["Race_Cat"], "White") or 0) - (pct(g["Race_Cat"], "Black") or 0) - (pct(g["Race_Cat"], "Asian") or 0),
            "pct_Democrat": pct(g["PartyCat"], "Democrat"),
            "pct_Republican": pct(g["PartyCat"], "Republican"),
            "pct_Independent": pct(g["PartyCat"], "Independent"),
            "pct_Extrem3": pct(g["Extremism"], 3.0),
        })
    all_row = {
        "Quintile": "All",
        "n": len(df),
        "IC_mean": df["IC"].mean(),
        "Ed_mean": df["Education_num"].mean(),
        "Ed_BA+": 100.0 * (df["Education_num"] >= 5).mean(),
        "Age_mean": df["AgeYears"].mean(),
        "pct_Female": pct(df["GenderCat"], "Female"),
        "pct_Male": pct(df["GenderCat"], "Male"),
        "pct_White": pct(df["Race_Cat"], "White"),
        "pct_Black": pct(df["Race_Cat"], "Black"),
        "pct_Asian": pct(df["Race_Cat"], "Asian"),
        "pct_OtherRace": 100.0 - (pct(df["Race_Cat"], "White") or 0) - (pct(df["Race_Cat"], "Black") or 0) - (pct(df["Race_Cat"], "Asian") or 0),
        "pct_Democrat": pct(df["PartyCat"], "Democrat"),
        "pct_Republican": pct(df["PartyCat"], "Republican"),
        "pct_Independent": pct(df["PartyCat"], "Independent"),
        "pct_Extrem3": pct(df["Extremism"], 3.0),
    }
    rows.append(all_row)
    return pd.DataFrame(rows)


def screening_operating_point(df, flag_quintile="Q1",
                              large_change_thresh=20.0,
                              adverse_thresh=-5.0,
                              weights=None):
    """Compute large-change retention and adverse captured given quintile flag.

    Large-change retention = fraction of participants with DeltaBelief >= 20
    whose IC is NOT in the flagged quintile (i.e., they would still have
    been served by the system without the adaptive safeguard).
    Adverse captured = fraction of participants with DeltaBelief <= -5 whose
    IC IS in the flagged quintile (the screen catches them for safeguard).
    """
    if weights is None:
        weights = np.ones(len(df))
    dv = df["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    flagged = (df["IC_q"] == flag_quintile).to_numpy()
    large = dv >= large_change_thresh
    adverse = dv <= adverse_thresh
    w_large = weights * large
    w_adv = weights * adverse
    if w_large.sum() > 0:
        retention = float(((~flagged) * w_large).sum() / w_large.sum()) * 100.0
    else:
        retention = np.nan
    if w_adv.sum() > 0:
        captured = float((flagged * w_adv).sum() / w_adv.sum()) * 100.0
    else:
        captured = np.nan
    return retention, captured


def education_reweighted_point(df, flag_quintile="Q1"):
    """Reweight each participant so that per-education-category weights sum
    to the same total across quintiles (removes quintile-specific
    over/under-representation of education levels).
    """
    s = df.dropna(subset=["Education_num"]).copy()
    ed_overall = s["Education_num"].value_counts(normalize=True)
    w = np.zeros(len(s))
    for q, g in s.groupby("IC_q", observed=True):
        ed_q = g["Education_num"].value_counts(normalize=True)
        for eid in g.index:
            e = s.loc[eid, "Education_num"]
            if e in ed_q and ed_q[e] > 0 and e in ed_overall:
                w[s.index.get_loc(eid)] = ed_overall[e] / ed_q[e]
    w = w * (len(s) / w.sum())
    s["w"] = w
    return screening_operating_point(s, flag_quintile=flag_quintile,
                                     weights=s["w"].to_numpy())


def fit_quadratic_with_interaction(df, moderator_col):
    """DV ~ IC + IC^2 + M + IC*M + IC^2*M + Pre_Belief + WordCount."""
    s = df.dropna(subset=["IC", "DV_BeliefChange_Specific", "Pre_Belief_Specific",
                          "OpenendedResponseWordCount", moderator_col]).copy()
    y = s["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    ic_raw = s["IC"].to_numpy(dtype=float)
    ic = zs(ic_raw)
    ic2 = zs(ic_raw ** 2)
    m = zs(s[moderator_col].to_numpy(dtype=float))
    pre = zs(s["Pre_Belief_Specific"].to_numpy(dtype=float))
    wc = zs(s["OpenendedResponseWordCount"].to_numpy(dtype=float))
    ic_m = ic * m
    ic2_m = ic2 * m
    X_full = sm.add_constant(np.column_stack([ic, ic2, m, ic_m, ic2_m, pre, wc]))
    m_full = sm.OLS(y, X_full).fit()
    X_red = sm.add_constant(np.column_stack([ic, ic2, m, pre, wc]))
    m_red = sm.OLS(y, X_red).fit()
    bf_joint = float(np.exp((m_red.bic - m_full.bic) / 2))
    return dict(
        n=len(s),
        b_ic_m=float(m_full.params[4]), p_ic_m=float(m_full.pvalues[4]),
        b_ic2_m=float(m_full.params[5]), p_ic2_m=float(m_full.pvalues[5]),
        bf_joint=bf_joint,
        b_ic2_full=float(m_full.params[2]), p_ic2_full=float(m_full.pvalues[2]),
    )


def main():
    df = load_df()
    print(f"N = {len(df)}; IC-quintile bin counts:")
    print(df["IC_q"].value_counts().sort_index())

    print("\n=== (a) Per-IC-quintile demographic composition ===")
    tbl = demographic_table(df)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(tbl.round(2).to_string(index=False))

    print("\n=== (b) Screening operating points ===")
    unw_r, unw_a = screening_operating_point(df, flag_quintile="Q1")
    print(f"  Unweighted (full sample)       : retention = {unw_r:.1f}%   adverse captured = {unw_a:.1f}%")
    ew_r, ew_a = education_reweighted_point(df, flag_quintile="Q1")
    print(f"  Education-stratum reweighted   : retention = {ew_r:.1f}%   adverse captured = {ew_a:.1f}%")

    print("\n=== (b2) Within each broad education band, retention / adverse-captured by Q1 screen ===")
    bands = {"HS-or-less": [1, 2], "SomeCollege/Assoc": [3, 4], "BA+": [5, 6, 7, 8]}
    for band, levels in bands.items():
        sub = df[df["Education_num"].isin(levels)].copy()
        r, a = screening_operating_point(sub, flag_quintile="Q1")
        flagged_rate = 100.0 * (sub["IC_q"] == "Q1").mean()
        print(f"  {band:<18}  n = {len(sub):>4}  flagged% = {flagged_rate:5.1f}  retention = {r:5.1f}%  adverse captured = {a:5.1f}%")

    print("\n=== (c) Fairness-sensitivity interactions ===")
    for mod in ["Education_num", "AgeYears"]:
        r = fit_quadratic_with_interaction(df, mod)
        print(
            f"  IC^2 x {mod:<15}  n = {r['n']:>4}   "
            f"b_IC*M  = {r['b_ic_m']:+.2f} (p={r['p_ic_m']:.3f})   "
            f"b_IC2*M = {r['b_ic2_m']:+.2f} (p={r['p_ic2_m']:.3f})   "
            f"BF_joint = {r['bf_joint']:.3f}"
        )


if __name__ == "__main__":
    main()
