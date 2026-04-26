#!/usr/bin/env python3
"""note23_incremental_validity.py — SI Note 23.

Tests whether the user-IC inverted-U is a re-labelling of more familiar
individual-difference variables: education, age, political extremism,
religion, generative-AI trust, or a verbal-fluency proxy (TTR).

Reproduces SI Table tab:incremental_validity:
  - Canonical (pre-belief + word count, full sample)
  - Canonical on full-covariate subsample (composition-only effect)
  - + Education + Age + Extremism + Religion + GenAI-trust (main-text headline)
  - + Education + Age + Extremism + TTR (verbal-fluency variant)

Inputs:
  data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv
  data/costello2024/analysis_data.csv
  data/costello2024/texts_for_scoring.jsonl
  data/costello2024/Data 8.28.24/AllDataForPublication.PPI.8.28.24.csv
    (raw Costello — needed for religion, genai_trust, AgeYears, Extremism)
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"
RAW = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"

ED_ORDER = {
    "LessThanHighSchool": 1, "HighSchool": 2, "SomeCollege": 3,
    "Associate": 4, "Bachelors": 5, "Masters": 6, "JD/MD": 7, "PhD": 8,
}


def zs(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def ttr(text):
    toks = re.findall(r"[A-Za-z']+", str(text).lower())
    return len(set(toks)) / max(len(toks), 1)


def fit_quadratic(df, extras=()):
    covars = ["Pre_Belief_Specific", "OpenendedResponseWordCount", *extras]
    s = df.dropna(subset=["IC", "DV_BeliefChange_Specific", *covars]).copy()
    y = s["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    raw = s["IC"].to_numpy(dtype=float)
    ic = zs(raw); ic2 = zs(raw ** 2)
    cov_z = [zs(s[c].to_numpy(dtype=float)) for c in covars]
    X1 = sm.add_constant(np.column_stack([ic, *cov_z]))
    X2 = sm.add_constant(np.column_stack([ic, ic2, *cov_z]))
    m1 = sm.OLS(y, X1).fit()
    m2 = sm.OLS(y, X2).fit()
    bf = float(np.exp((m1.bic - m2.bic) / 2))
    sig1, sig2 = float(np.std(raw, ddof=0)), float(np.std(raw ** 2, ddof=0))
    b_lin, b_q = float(m2.params[1]), float(m2.params[2])
    apex = -b_lin * sig2 / (2 * b_q * sig1) if b_q != 0 else np.nan
    return dict(n=len(s), b_lin=b_lin, p_lin=float(m2.pvalues[1]),
                b_q=b_q, p_q=float(m2.pvalues[2]), BF10=bf, apex=apex)


def main():
    if not RAW.exists():
        from _raw_data_check import require_raw
        require_raw(RAW, "Costello (raw publication CSV)", "https://osf.io/gdkb7/")

    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    ic = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ad = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)
    raw = pd.read_csv(RAW, low_memory=False).drop_duplicates("participantId", keep="first")

    base = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "text_initial":  [m.get("text_initial", "") for m in meta],
        "IC":             ic["ic_qwenorpo400_logit"].astype(float).values,
    })
    base["TTR"] = base["text_initial"].apply(ttr)

    raw_keep = ["participantId", "AgeYears", "Education_Cat", "Extremism",
                "religion", "genai_trust"]
    raw_keep = [c for c in raw_keep if c in raw.columns]
    rd = raw[raw_keep].copy()
    if "Education_Cat" in rd.columns:
        rd["Education_num"] = rd["Education_Cat"].map(ED_ORDER).astype(float)
    for c in ["AgeYears", "Extremism", "religion", "genai_trust"]:
        if c in rd.columns:
            rd[c] = pd.to_numeric(rd[c], errors="coerce")

    df = (base.merge(ad[["participantId", "DV_BeliefChange_Specific",
                         "Pre_Belief_Specific", "OpenendedResponseWordCount"]],
                     on="participantId", how="inner")
              .merge(rd, on="participantId", how="left"))

    print("SI Note 23 — Incremental validity of beta_IC^2 beyond demographics + verbal fluency\n")

    full_demo = ["Education_num", "AgeYears", "Extremism", "religion", "genai_trust"]
    full_ttr  = ["Education_num", "AgeYears", "Extremism", "TTR"]

    print(f"{'Model':<60}{'n':>6}  {'beta_IC^2':>10}  {'p':>10}  {'BF10':>10}  {'apex':>6}")

    r0 = fit_quadratic(df)
    print(f"  {'Canonical (pre-belief + word count, full sample)':<58}"
          f"{r0['n']:>6}  {r0['b_q']:>+10.3f}  {r0['p_q']:>10.2g}  {r0['BF10']:>10.1f}  {r0['apex']:>+6.2f}")

    df_demo = df.dropna(subset=full_demo + ["IC","DV_BeliefChange_Specific",
                                             "Pre_Belief_Specific","OpenendedResponseWordCount"])
    r_match_demo = fit_quadratic(df_demo)
    print(f"  {'Canonical, full-covariate subsample (demographic-set composition)':<58}"
          f"{r_match_demo['n']:>6}  {r_match_demo['b_q']:>+10.3f}  {r_match_demo['p_q']:>10.2g}  "
          f"{r_match_demo['BF10']:>10.1f}  {r_match_demo['apex']:>+6.2f}")

    r_demo = fit_quadratic(df, extras=tuple(full_demo))
    print(f"  {'+ Age + Education + Extremism + Religion + GenAI-trust (headline)':<58}"
          f"{r_demo['n']:>6}  {r_demo['b_q']:>+10.3f}  {r_demo['p_q']:>10.2g}  "
          f"{r_demo['BF10']:>10.1f}  {r_demo['apex']:>+6.2f}")

    df_ttr = df.dropna(subset=full_ttr + ["IC","DV_BeliefChange_Specific",
                                           "Pre_Belief_Specific","OpenendedResponseWordCount"])
    r_match_ttr = fit_quadratic(df_ttr)
    print(f"  {'Canonical, full-covariate subsample (TTR-variant composition)':<58}"
          f"{r_match_ttr['n']:>6}  {r_match_ttr['b_q']:>+10.3f}  {r_match_ttr['p_q']:>10.2g}  "
          f"{r_match_ttr['BF10']:>10.1f}  {r_match_ttr['apex']:>+6.2f}")

    r_ttr = fit_quadratic(df, extras=tuple(full_ttr))
    print(f"  {'+ Age + Education + Extremism + TTR (verbal-fluency variant)':<58}"
          f"{r_ttr['n']:>6}  {r_ttr['b_q']:>+10.3f}  {r_ttr['p_q']:>10.2g}  "
          f"{r_ttr['BF10']:>10.1f}  {r_ttr['apex']:>+6.2f}")


if __name__ == "__main__":
    main()
