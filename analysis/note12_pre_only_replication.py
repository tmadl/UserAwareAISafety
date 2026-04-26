#!/usr/bin/env python3
"""note12_pre_only_replication.py — SI Note 12.

Reproduces the pre-treatment-only IC replication tables for Costello and
Boissin: same downstream moderation effect when IC is scored from pre-treatment
text only, ruling out engagement-during-dialogue contamination as the source
of the moderation signal.

Costello (tab in Note 12):
  Q400 pre-only (primary):           beta_IC^2 = -15.17,  BF = 1086,  apex = 2.76
  gpt-4.1-mini pre-only:             beta_IC^2 = -14.24,  BF = 487k,  apex = 3.12
  gpt-4.1-mini full-dialogue:        beta_IC^2 = -18.15,  BF =  69k,  apex = 3.75

Boissin (tab in Note 12):
  Q400 text_all (full; primary):   β_lin = +1.60, p = .035
  Q400 text_initial (pre-only):    β_lin = +0.57, p = .476
  AI × Human-like, full:           β_lin = +3.11, p = .024
  AI × Human-like, pre-only:       β_lin = +1.43, p = .31
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_Q = DATA / "ic_qwen3orpo400"


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def costello_quad(df, ic_col):
    """Canonical Costello quadratic + bootstrap apex CI."""
    sub = df.dropna(subset=["DV_BeliefChange_Specific", ic_col,
                            "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()
    y = sub["DV_BeliefChange_Specific"].values.astype(float)
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
    # Bootstrap apex CI (2,000 reps)
    rng = np.random.default_rng(42)
    n = len(sub); peaks = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        rb = raw[idx]
        Xb = sm.add_constant(np.column_stack([zs(rb), zs(rb ** 2),
                                              zs(sub["Pre_Belief_Specific"].values[idx]),
                                              zs(sub["OpenendedResponseWordCount"].values[idx])]))
        try:
            mb = sm.OLS(y[idx], Xb).fit()
            sb_ic, sb_sq = np.std(rb), np.std(rb ** 2)
            peaks.append(-mb.params[1] * sb_sq / (2 * mb.params[2] * sb_ic))
        except Exception:
            pass
    lo, hi = np.percentile(peaks, [2.5, 97.5])
    return dict(n=len(sub), beta=m_quad.params[2], p=m_quad.pvalues[2],
                bf=bf, apex=apex, ci_lo=lo, ci_hi=hi)


def boissin_lin(df, ic_col, subset=None):
    """Boissin linear: belief_change ~ IC + PreBelief (raw scale)."""
    sub = df.dropna(subset=["belief_change", ic_col, "PreBelief"])
    if subset is not None:
        sub = sub[subset.loc[sub.index]]
    y = sub["belief_change"].values.astype(float)
    ic = sub[ic_col].values.astype(float)
    pre = sub["PreBelief"].values.astype(float)
    X_full = sm.add_constant(np.column_stack([ic, pre]))
    X_red = sm.add_constant(pre.reshape(-1, 1))
    m_full = sm.OLS(y, X_full).fit()
    m_red = sm.OLS(y, X_red).fit()
    bf_lin_null = float(np.exp((m_red.bic - m_full.bic) / 2))
    return dict(n=len(sub), beta=m_full.params[1], p=m_full.pvalues[1],
                bf_lin_null=bf_lin_null)


def main():
    # === Costello ===
    print("=== Costello — pre-treatment-only IC replication ===")
    an = pd.read_csv(DATA / "costello2024" / "analysis_data.csv")
    ic_old = pd.read_csv(DATA / "costello2024" / "all_complexity_scores.csv")
    meta = [json.loads(l) for l in open(DATA / "costello2024" / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic_q400 = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC_q400_pre": q["ic_qwenorpo400_logit"].astype(float).values,
    })
    df = an.merge(ic_old, on="participantId", how="inner").merge(ic_q400, on="participantId", how="left")
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific", "OpenendedResponseWordCount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"\n{'Configuration':<45s} {'n':>5s} {'β_IC²':>8s} {'p':>8s} {'BF₁₀':>10s} {'apex':>6s} {'95% CI':>14s}")
    print("-" * 100)
    for label, col in [("Q400 pre-only (primary)",            "IC_q400_pre"),
                       ("gpt-4.1-mini pre-only",              "IC_openai_initial"),
                       ("gpt-4.1-mini full-dialogue",         "IC_openai")]:
        r = costello_quad(df, col)
        print(f"{label:<45s} {r['n']:>5d} {r['beta']:>+8.2f} {fmt_p(r['p']):>8s} {r['bf']:>10.0f} "
              f"{r['apex']:>6.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    # Cross-source correlations
    print("\nCross-source IC correlations (Costello):")
    sub = df.dropna(subset=["IC_q400_pre", "IC_openai_initial"])
    r1, _ = sp.pearsonr(sub["IC_q400_pre"], sub["IC_openai_initial"])
    sub = df.dropna(subset=["IC_openai_initial", "IC_openai"])
    r2, _ = sp.pearsonr(sub["IC_openai_initial"], sub["IC_openai"])
    print(f"  r(Q400 pre-only, gpt pre-only)         = {r1:.3f}")
    print(f"  r(gpt pre-only, gpt full-dialogue)     = {r2:.3f}")

    # === Boissin ===
    print("\n=== Boissin — pre-treatment-only IC replication ===")
    an_b = pd.read_csv(DATA / "boissin2025" / "analysis_data.csv")
    rows = [json.loads(l) for l in open(DATA_Q / "boissin_texts_for_scoring_qwenorpo400.jsonl")]
    ic_b = pd.DataFrame({
        "participantId": [r["participantId"] for r in rows],
        "IC_q400_initial": [r["ic_qwenorpo400_initial_logit"] for r in rows],
        "IC_q400_all":     [r["ic_qwenorpo400_all_logit"] for r in rows],
    })
    df_b = an_b.merge(ic_b, on="participantId", how="left")
    for c in ["belief_change", "PreBelief", "IC_q400_initial", "IC_q400_all"]:
        df_b[c] = pd.to_numeric(df_b[c], errors="coerce")

    print(f"\n{'Configuration':<48s} {'n':>5s} {'β_lin':>8s} {'p':>8s} {'BF(lin/null)':>13s}")
    print("-" * 90)
    for label, col in [("Q400 text_all (full; primary)", "IC_q400_all"),
                       ("Q400 text_initial (pre-only)",  "IC_q400_initial")]:
        r = boissin_lin(df_b, col)
        print(f"{label:<48s} {r['n']:>5d} {r['beta']:>+8.2f} {fmt_p(r['p']):>8s} {r['bf_lin_null']:>13.2f}")

    # AI × Human-like cell
    print("\nAI × Human-like cell:")
    aihl = df_b[(df_b["Speaker"] == "AI") & (df_b["PromptType"] == "Human-like")].copy()
    for label, col in [("full-dialogue", "IC_q400_all"),
                       ("pre-only",      "IC_q400_initial")]:
        r = boissin_lin(aihl, col)
        print(f"  {label:<14s} n = {r['n']:>4d}  β_lin = {r['beta']:>+5.2f}, p = {fmt_p(r['p']):>6s}")

    # Cross-source r within Boissin
    sub = df_b.dropna(subset=["IC_q400_initial", "IC_q400_all"])
    r3, _ = sp.pearsonr(sub["IC_q400_initial"], sub["IC_q400_all"])
    print(f"\n  r(Q400 pre-only, Q400 full-dialogue) on same participants = {r3:.3f}")


if __name__ == "__main__":
    main()
