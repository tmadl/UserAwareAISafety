#!/usr/bin/env python3
"""note18_turnwise_ic_stability.py — SI Note 18.

Per-turn IC scoring across the four user turns of the Costello dialogue plus
cumulative concatenations (cum_1 = turn_0 alone, cum_4 = full dialogue).
Reproduces tab:turnwise_descriptives (per-turn word counts, mean IC, SD)
and tab:turnwise_fits (inverted-U β, BF, apex per scoring window) plus
within-person reliability (ICC) numbers.
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


def fit(df, ic_col):
    """Canonical Costello quadratic on a particular IC scoring window."""
    need = [ic_col, "DV_BeliefChange_Specific", "Pre_Belief_Specific",
            "OpenendedResponseWordCount"]
    s = df[need].dropna().copy()
    y = s["DV_BeliefChange_Specific"].values.astype(float)
    raw = s[ic_col].values.astype(float)
    ic_z, ic2_z = zs(raw), zs(raw ** 2)
    pre = zs(s["Pre_Belief_Specific"].values)
    wc = zs(s["OpenendedResponseWordCount"].values)
    X_lin = sm.add_constant(np.column_stack([ic_z, pre, wc]))
    X_quad = sm.add_constant(np.column_stack([ic_z, ic2_z, pre, wc]))
    m_lin = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()
    bf = float(np.exp((m_lin.bic - m_quad.bic) / 2))
    sd_ic, sd_sq = np.std(raw), np.std(raw ** 2)
    b1, b2 = m_quad.params[1], m_quad.params[2]
    apex = -b1 * sd_sq / (2 * b2 * sd_ic) if abs(b2) > 1e-12 else np.nan
    return dict(n=len(s), beta=m_quad.params[2], p=m_quad.pvalues[2],
                bf=bf, apex=apex)


def icc_3_1(matrix):
    """Two-way mixed, single-rater consistency ICC across columns."""
    n, k = matrix.shape
    grand = matrix.mean()
    row_means, col_means = matrix.mean(axis=1), matrix.mean(axis=0)
    SS_total = ((matrix - grand) ** 2).sum()
    SS_row = k * ((row_means - grand) ** 2).sum()
    SS_col = n * ((col_means - grand) ** 2).sum()
    SS_err = SS_total - SS_row - SS_col
    MS_row = SS_row / (n - 1)
    MS_err = SS_err / ((n - 1) * (k - 1))
    return (MS_row - MS_err) / (MS_row + (k - 1) * MS_err)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def main():
    turns = pd.read_csv(DATA / "texts_for_scoring_all_qwenorpo400_turns.csv")
    init = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ad = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount"]:
        ad[c] = pd.to_numeric(ad[c], errors="coerce")

    # Match turns row → analysis_data row by first-segment text
    # (`Paragraph` column in turns CSV contains the full dialogue with `|||` separators;
    # `Paragraph` in the initial CSV contains text_initial alone)
    turns["turn_0_text"] = (turns["Paragraph"].astype(str)
                            .str.split(r"\|\|\|", regex=True, n=1).str[0].str.strip())
    init["init_text"] = init["Paragraph"].astype(str).str.strip()
    text_to_idx = {t: i for i, t in enumerate(init["init_text"])}
    turns["ad_idx"] = turns["turn_0_text"].map(text_to_idx)
    assert turns["ad_idx"].notna().all(), "unmatched rows in turns merge"
    merged = turns.merge(
        ad[["DV_BeliefChange_Specific", "Pre_Belief_Specific",
            "OpenendedResponseWordCount"]].reset_index().rename(columns={"index": "ad_idx"}),
        on="ad_idx", how="left",
    )

    # Per-turn descriptive table (tab:turnwise_descriptives)
    print("SI Note 18 — Turnwise IC scoring (Costello within-dialogue subset)\n")
    parts = merged["Paragraph"].apply(lambda t: [p.strip() for p in str(t).split("|||")])
    print(f"{'Turn':<35s} {'n':>5s} {'wc median':>10s} {'mean IC':>9s} {'SD IC':>7s}")
    print("-" * 70)
    for k, label in enumerate(["turn_0 (initial belief statement)",
                               "turn_1 (first reactive response)",
                               "turn_2", "turn_3"]):
        ic_col = f"turn_{k}_ic_logit"
        ic_vals = merged[ic_col].dropna()
        wcs = pd.Series([len(p[k].split()) if k < len(p) else np.nan
                         for p in parts]).dropna()
        print(f"{label:<35s} {len(ic_vals):>5d} {wcs.median():>10.0f} "
              f"{ic_vals.mean():>9.2f} {ic_vals.std():>7.2f}")

    # Within-person reliability
    tcols = [f"turn_{k}_ic_logit" for k in range(4)]
    ccols = [f"cum_{k}_ic_logit" for k in range(1, 5)]
    tmat = merged[tcols].dropna().values
    cmat = merged[ccols].dropna().values
    print(f"\nWithin-person reliability:")
    print(f"  ICC(3,1) across 4 per-turn scores  = {icc_3_1(tmat):.2f} (n = {len(tmat)})")
    print(f"  ICC(3,1) across cum_1 → cum_4      = {icc_3_1(cmat):.2f} (n = {len(cmat)})")

    # Inverted-U refits (tab:turnwise_fits)
    print("\nInverted-U refit per IC scoring window:")
    print(f"{'IC source':<35s} {'n':>5s}  {'β_IC²':>8s}  {'p':>8s}  {'BF₁₀':>8s}  {'apex':>5s}")
    print("-" * 80)

    # Reference row: full-sample pre-only
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ref = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC": q["ic_qwenorpo400_logit"].astype(float).values,
    })
    full = ad.merge(ref, on="participantId", how="inner")
    r = fit(full, "IC")
    print(f"{'(reference) full N pre-only':<35s} {r['n']:>5d}  {r['beta']:>+8.2f}  "
          f"{fmt_p(r['p']):>8s}  {r['bf']:>8.1f}  {r['apex']:>5.2f}")
    print()

    for col in tcols:
        r = fit(merged, col)
        apex_str = f"{r['apex']:>5.2f}" if not np.isnan(r['apex']) else "  ---"
        # Apex is meaningless when curvature non-significant
        if r['p'] > 0.1:
            apex_str = "  ---"
        print(f"{col:<35s} {r['n']:>5d}  {r['beta']:>+8.2f}  "
              f"{fmt_p(r['p']):>8s}  {r['bf']:>8.2f}  {apex_str}")
    print()
    for col in ccols:
        r = fit(merged, col)
        apex_str = f"{r['apex']:>5.2f}" if not np.isnan(r['apex']) else "  ---"
        if r['p'] > 0.1:
            apex_str = "  ---"
        print(f"{col:<35s} {r['n']:>5d}  {r['beta']:>+8.2f}  "
              f"{fmt_p(r['p']):>8s}  {r['bf']:>8.2f}  {apex_str}")


if __name__ == "__main__":
    main()
