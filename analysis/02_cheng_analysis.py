#!/usr/bin/env python3
"""
02_cheng_analysis.py — Cheng et al. reanalysis (sycophantic AI).

Reproduces the paper's core Cheng boundary-condition analysis:
  1. Pooled linear IC effect on perceived rightness
  2. Quadratic-vs-linear BIC / BF10 (predicted: favours linear)
  3. Within-cell (Study × sycophancy) linear fits
  4. Condition interaction model
  5. Polarization distance from midpoint (Study 3)

Primary scorer: Qwen3-ORPO-400 logit-EV on text_initial
Cross-scorer:   gpt-4.1-mini on same pre-treatment text (IC_openai_initial)
"""

import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cheng2006"
DATA_Q400 = ROOT / "data" / "ic_qwen3orpo400"


def zs(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def bf10_from_bic(bic_reduced, bic_full):
    """BF10 favouring full over reduced from BIC difference."""
    return np.exp((bic_reduced - bic_full) / 2.0)


def load_data():
    """Primary: Q400 logit-EV, merged via texts_for_scoring.jsonl row index.
    Cross-scorer: IC_openai_initial (gpt-4.1-mini on pre-treatment text)."""
    an = pd.read_csv(DATA / "analysis_data.csv")

    # Primary Q400 logit-EV
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q400 = pd.read_csv(DATA_Q400 /
                       "cheng_texts_for_scoring_initial_qwenorpo400.csv")
    assert len(meta) == len(q400), f"{len(meta)} vs {len(q400)}"
    ic_q400 = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "ic_q400": q400["ic_qwenorpo400_logit"].astype(float).values,
    })

    # Cross-scorer (gpt-4.1-mini). Cheng uses the conversation-level IC_openai
    # (whole-dialogue scoring) rather than pre-treatment text, because
    # participants' pre-text in Cheng is a brief conflict setup; the paper's
    # reported cross-scorer numbers (β = -0.303, p < .001) are on IC_openai.
    ic_old = pd.read_csv(DATA / "all_complexity_scores.csv")[
        ["participantId", "IC_openai", "IC_openai_initial"]]
    ic_old = ic_old.rename(columns={
        "IC_openai": "ic_openai_conv",
        "IC_openai_initial": "ic_openai_pre"})

    df = an.merge(ic_q400, on="participantId", how="left")
    df = df.merge(ic_old, on="participantId", how="left")

    for c in ["rightorwrong", "repair_score", "trust_score", "wc_all",
              "wc_initial", "age", "is_male", "is_sycophantic", "study"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ── Analysis 1: Pooled linear + quadratic-vs-linear BF ────────────────────

def analysis_pooled_linear(df, ic_col, label):
    """Pooled linear IC effect + quadratic-vs-linear BIC comparison."""
    print(f"\n{'='*70}")
    print(f"  POOLED IC → rightorwrong — {label}")
    print(f"{'='*70}")

    dv = "rightorwrong"
    sub = df.dropna(subset=[dv, ic_col]).copy()
    y = sub[dv].values

    ic = zs(sub[ic_col].values)
    ic_sq = zs(sub[ic_col].values ** 2)

    X_lin = sm.add_constant(np.column_stack([ic]))
    X_quad = sm.add_constant(np.column_stack([ic, ic_sq]))
    m_lin = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()

    bf = bf10_from_bic(m_lin.bic, m_quad.bic)  # quadratic over linear
    print(f"\n  n = {len(sub)}")
    print(f"  Linear:    β_IC = {m_lin.params[1]:+.3f}, p = {fmt_p(m_lin.pvalues[1])}, R² = {m_lin.rsquared:.4f}")
    print(f"  Quadratic: β_IC = {m_quad.params[1]:+.3f} (p = {fmt_p(m_quad.pvalues[1])}), "
          f"β_IC² = {m_quad.params[2]:+.3f} (p = {fmt_p(m_quad.pvalues[2])})")
    print(f"  BF10 (quadratic vs linear, BIC-based) = {bf:.3f}")

    return {
        "n": len(sub),
        "beta_lin": m_lin.params[1], "p_lin": m_lin.pvalues[1],
        "beta_ic2": m_quad.params[2], "p_ic2": m_quad.pvalues[2],
        "bf_quad_lin": bf,
    }


# ── Analysis 2: Within-cell linear (Study × sycophancy) ────────────────────

def analysis_within_cell(df, ic_col, label):
    """Linear IC effect in each of the four Study × sycophancy cells."""
    print(f"\n{'='*70}")
    print(f"  WITHIN-CELL LINEAR — {label}")
    print(f"{'='*70}")

    dv = "rightorwrong"
    print(f"\n  {'Cell':<22s} {'n':>5s} {'β_lin':>9s} {'p':>8s} {'BF10 lin>null':>14s}")
    print(f"  {'-'*62}")

    results = {}
    for study in [2, 3]:
        for syco in [1, 0]:
            mask = (df["study"] == study) & (df["is_sycophantic"] == syco)
            sub = df[mask].dropna(subset=[dv, ic_col]).copy()
            if len(sub) < 30:
                continue
            y = sub[dv].values
            ic = zs(sub[ic_col].values)

            X0 = sm.add_constant(np.ones_like(ic).reshape(-1, 1))  # intercept-only
            # intercept-only model
            m0 = sm.OLS(y, np.ones_like(ic).reshape(-1, 1)).fit()
            m1 = sm.OLS(y, sm.add_constant(ic)).fit()
            bf = bf10_from_bic(m0.bic, m1.bic)

            name = f"S{study} {'syco' if syco else 'non-syco'}"
            print(f"  {name:<22s} {len(sub):>5d} {m1.params[1]:>+9.3f} "
                  f"{fmt_p(m1.pvalues[1]):>8s} {bf:>14.2f}")
            results[name] = {"n": len(sub), "beta_lin": m1.params[1],
                             "p": m1.pvalues[1], "bf": bf}
    return results


# ── Analysis 3: Interaction (condition × IC) ──────────────────────────────

def analysis_interaction(df, ic_col, label):
    """Condition × IC interaction model (pooled)."""
    print(f"\n{'='*70}")
    print(f"  CONDITION × IC INTERACTION — {label}")
    print(f"{'='*70}")

    dv = "rightorwrong"
    sub = df.dropna(subset=[dv, ic_col, "is_sycophantic"]).copy()
    y = sub[dv].values
    ic = zs(sub[ic_col].values)
    ic_sq = zs(sub[ic_col].values ** 2)
    cond = sub["is_sycophantic"].values.astype(float)

    X = sm.add_constant(np.column_stack([ic, ic_sq, cond, ic * cond, ic_sq * cond]))
    m = sm.OLS(y, X).fit()

    names = ["const", ic_col, f"{ic_col}²", "sycophantic",
             f"{ic_col}×syco", f"{ic_col}²×syco"]
    print(f"\n  n = {len(sub)}, R² = {m.rsquared:.4f}")
    print(f"\n  {'Term':<25s} {'β':>8s} {'SE':>8s} {'p':>8s}")
    print(f"  {'-'*55}")
    for i, name in enumerate(names):
        print(f"  {name:<25s} {m.params[i]:>+8.3f} {m.bse[i]:>8.3f} {fmt_p(m.pvalues[i]):>8s}")


# ── Analysis 4: Polarization distance (Study 3 only) ──────────────────────

def analysis_polarization(df, ic_col, label):
    """|rightorwrong - 4| on 7-point scale (Study 3)."""
    print(f"\n{'='*70}")
    print(f"  POLARIZATION DIST (|post - 4|, Study 3) — {label}")
    print(f"{'='*70}")

    dv = "rightorwrong"
    sub = df[df["study"] == 3].dropna(subset=[dv, ic_col, "is_sycophantic"]).copy()
    sub["pol_dist"] = (sub[dv] - 4.0).abs()

    for cond_val, cond_name in [(1, "sycophantic"), (0, "non-sycophantic")]:
        s = sub[sub["is_sycophantic"] == cond_val]
        if len(s) < 30:
            continue
        y = s["pol_dist"].values
        ic = zs(s[ic_col].values)
        ic_sq = zs(s[ic_col].values ** 2)
        X_lin = sm.add_constant(np.column_stack([ic]))
        X_quad = sm.add_constant(np.column_stack([ic, ic_sq]))
        m_lin = sm.OLS(y, X_lin).fit()
        m_quad = sm.OLS(y, X_quad).fit()
        print(f"  {cond_name:<16s} n={len(s):>4d}  "
              f"β_IC = {m_lin.params[1]:+.3f} (p = {fmt_p(m_lin.pvalues[1])}), "
              f"β_IC² = {m_quad.params[2]:+.3f} (p = {fmt_p(m_quad.pvalues[2])})")

    # Joint interaction F
    y = sub["pol_dist"].values
    ic = zs(sub[ic_col].values)
    ic_sq = zs(sub[ic_col].values ** 2)
    cond = sub["is_sycophantic"].values.astype(float)
    X_null = sm.add_constant(np.column_stack([cond]))
    X_full = sm.add_constant(np.column_stack([ic, ic_sq, cond, ic * cond, ic_sq * cond]))
    m0 = sm.OLS(y, X_null).fit()
    m1 = sm.OLS(y, X_full).fit()
    f_val = ((m0.ssr - m1.ssr) / (m1.df_model - m0.df_model)) / (m1.ssr / m1.df_resid)
    p_val = sp.f.sf(f_val, m1.df_model - m0.df_model, m1.df_resid)
    print(f"  Joint cond×IC polarization F({int(m1.df_model - m0.df_model)}, "
          f"{int(m1.df_resid)}) = {f_val:.2f}, p = {fmt_p(p_val)}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    print(f"Loaded {len(df)} participants (analysis_data)")

    sub = df.dropna(subset=["ic_q400", "ic_openai_conv"])
    r, p = sp.pearsonr(sub["ic_q400"], sub["ic_openai_conv"])
    print(f"\nic_q400 (pre-text) vs ic_openai_conv (whole-dialogue): "
          f"r = {r:.3f}, p = {fmt_p(p)}, n = {len(sub)}")
    print(f"ic_q400         mean = {sub['ic_q400'].mean():.2f}, SD = {sub['ic_q400'].std():.2f}")
    print(f"ic_openai_conv  mean = {sub['ic_openai_conv'].mean():.2f}, SD = {sub['ic_openai_conv'].std():.2f}")

    for ic_col, label in [("ic_q400", "Primary (Q400 logit-EV, pre-text)"),
                          ("ic_openai_conv", "Cross-scorer (gpt-4.1-mini, whole-dialogue)")]:
        analysis_pooled_linear(df, ic_col, label)
        analysis_within_cell(df, ic_col, label)
        analysis_interaction(df, ic_col, label)
        analysis_polarization(df, ic_col, label)


if __name__ == "__main__":
    main()
