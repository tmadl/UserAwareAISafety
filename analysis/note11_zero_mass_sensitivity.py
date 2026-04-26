#!/usr/bin/env python3
"""note11_zero_mass_sensitivity.py — SI Note 11.

Reproduces SI Tables tab:zero_mass, tab:costello_movers, tab:boissin_movers
plus the Boissin AI × Human-like cell among-movers paragraph numbers.

Tests whether the inverted-U (Costello) and linear-monotone (Boissin) signals
survive subsetting to participants who actually moved (|Δ| > t) and, for
Boissin, direction-conditional subsetting.
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_Q = DATA / "ic_qwen3orpo400"


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}" if p >= .001 else f"{p:.4f}"


def load_costello():
    an = pd.read_csv(DATA / "costello2024" / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DATA / "costello2024" / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic_q = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC": q["ic_qwenorpo400_logit"].astype(float).values,
    })
    df = an.merge(ic_q, on="participantId", how="left")
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "IC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["DV_BeliefChange_Specific", "IC",
                              "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()


def load_boissin():
    an = pd.read_csv(DATA / "boissin2025" / "analysis_data.csv")
    rows = [json.loads(l) for l in open(DATA_Q / "boissin_texts_for_scoring_qwenorpo400.jsonl")]
    ic = pd.DataFrame({
        "participantId": [r["participantId"] for r in rows],
        "IC": [r["ic_qwenorpo400_all_logit"] for r in rows],  # full-dialogue (Boissin primary)
    })
    df = an.merge(ic, on="participantId", how="left")
    for c in ["belief_change", "PreBelief", "IC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["belief_change", "IC", "PreBelief"]).copy()


def quad_fit(sub, dv_col, ic_col):
    """Canonical Costello quadratic on a subset. Returns β_IC², p, BF, apex(raw)."""
    y = sub[dv_col].values.astype(float)
    raw = sub[ic_col].values.astype(float)
    ic_z, ic2_z = zs(raw), zs(raw ** 2)
    pre = zs(sub["Pre_Belief_Specific"].values)
    wc = zs(sub["OpenendedResponseWordCount"].values)
    X_lin = sm.add_constant(np.column_stack([ic_z, pre, wc]))
    X_quad = sm.add_constant(np.column_stack([ic_z, ic2_z, pre, wc]))
    m_lin = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()
    bf = float(np.exp((m_lin.bic - m_quad.bic) / 2))
    sd_ic, sd_sq = np.std(raw), np.std(raw ** 2)
    b1, b2 = m_quad.params[1], m_quad.params[2]
    apex = -b1 * sd_sq / (2 * b2 * sd_ic) if abs(b2) > 1e-12 else np.nan
    return dict(n=len(sub), beta=m_quad.params[2], p=m_quad.pvalues[2],
                bf=bf, apex=apex)


def boissin_lin(sub):
    """Boissin linear: belief_change ~ IC + PreBelief, IC and PreBelief both raw scale
    (paper convention for Boissin — see SI Note 12)."""
    y = sub["belief_change"].values.astype(float)
    ic = sub["IC"].values.astype(float)
    pre = sub["PreBelief"].values.astype(float)
    X = sm.add_constant(np.column_stack([ic, pre]))
    m = sm.OLS(y, X).fit()
    return dict(n=len(sub), beta=m.params[1], p=m.pvalues[1])


def main():
    cos = load_costello()
    boi = load_boissin()
    print(f"Loaded Costello n={len(cos)}; Boissin n={len(boi)}\n")

    # === 1. Zero-mass distribution table (tab:zero_mass) ===
    print("=== Table tab:zero_mass — zero-mass distribution ===")
    print(f"{'Dataset':<10s} {'n':>6s}  {'exact 0':>8s}  {'|Δ|<5':>8s}  {'|Δ|<10':>8s}")
    for label, df, dv_col in [("Costello", cos, "DV_BeliefChange_Specific"),
                              ("Boissin",  boi, "belief_change")]:
        d = df[dv_col].abs()
        print(f"{label:<10s} {len(df):>6d}  "
              f"{(df[dv_col]==0).mean()*100:>7.1f}%  "
              f"{(d<5).mean()*100:>7.1f}%  "
              f"{(d<10).mean()*100:>7.1f}%")

    # === 2. Costello |Δ|-threshold quadratic (tab:costello_movers) ===
    print("\n=== Table tab:costello_movers — Costello quadratic by |Δ| threshold ===")
    print(f"{'Exclusion':<14s} {'n':>5s}  {'% kept':>7s}  {'β_IC²':>8s}  {'p':>8s}  {'BF₁₀':>8s}  {'apex':>5s}")
    full_n = len(cos)
    for label, mask in [("None (full)", np.ones(len(cos), bool)),
                        ("|Δ| > 0",     cos["DV_BeliefChange_Specific"].abs() > 0),
                        ("|Δ| > 1",     cos["DV_BeliefChange_Specific"].abs() > 1),
                        ("|Δ| > 2",     cos["DV_BeliefChange_Specific"].abs() > 2),
                        ("|Δ| > 5",     cos["DV_BeliefChange_Specific"].abs() > 5),
                        ("|Δ| > 10",    cos["DV_BeliefChange_Specific"].abs() > 10)]:
        sub = cos[mask].copy()
        r = quad_fit(sub, "DV_BeliefChange_Specific", "IC")
        pct = 100 * len(sub) / full_n
        print(f"{label:<14s} {r['n']:>5d}  {pct:>6.1f}%  "
              f"{r['beta']:>+8.2f}  {fmt_p(r['p']):>8s}  {r['bf']:>8.1f}  {r['apex']:>5.2f}")

    # === 3. Boissin signed-DV linear by movement subset (tab:boissin_movers) ===
    print("\n=== Table tab:boissin_movers — Boissin linear by movement subset ===")
    print(f"{'Subset':<32s} {'n':>5s}  {'β_lin':>8s}  {'p':>8s}")
    boi_dv = boi["belief_change"]
    for label, mask in [("Full",                          np.ones(len(boi), bool)),
                        ("|Δ| > 0",                       boi_dv.abs() > 0),
                        ("|Δ| ≥ 5",                       boi_dv.abs() >= 5),
                        ("Δ < 0 (toward debunking)",      boi_dv < 0),
                        ("Δ > 0 (backfired)",             boi_dv > 0)]:
        sub = boi[mask].copy()
        r = boissin_lin(sub)
        print(f"{label:<32s} {r['n']:>5d}  {r['beta']:>+8.2f}  {fmt_p(r['p']):>8s}")

    # === 4. Boissin AI × Human-like cell among-movers paragraph ===
    print("\n=== AI × Human-like cell (Boissin Note 11 paragraph) ===")
    aihl = boi[(boi["Speaker"] == "AI") & (boi["PromptType"] == "Human-like")].copy()
    print(f"AI × Human-like cell n = {len(aihl)}")
    for label, mask in [("Full",       np.ones(len(aihl), bool)),
                        ("|Δ| > 0",    aihl["belief_change"].abs() > 0),
                        ("|Δ| ≥ 5",    aihl["belief_change"].abs() >= 5)]:
        sub = aihl[mask].copy()
        if len(sub) < 30:
            continue
        r = boissin_lin(sub)
        print(f"  {label:<10s} n = {r['n']:>4d}  β_lin = {r['beta']:>+5.2f}, p = {fmt_p(r['p']):>6s}")


if __name__ == "__main__":
    main()
