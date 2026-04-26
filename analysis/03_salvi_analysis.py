#!/usr/bin/env python3
"""
03_salvi_analysis.py — Salvi et al. reanalysis (adversarial debate).

Reproduces the paper's core analyses for Salvi data:
  1. IC moderation of opinion change (predicted null — boundary condition)
  2. Condition-specific models (AI vs human opponent)
  3. IC prediction of absolute opinion change magnitude

Runs the primary Q400 logit-EV scorer and the gpt-4.1-mini cross-scorer baseline.
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
DATA = ROOT / "data" / "salvi2005"
DATA_Q400 = ROOT / "data" / "ic_qwen3orpo400"


def zs(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def delta_r2_test(y, X_reduced, X_full):
    m1 = sm.OLS(y, X_reduced).fit()
    m2 = sm.OLS(y, X_full).fit()
    df_diff = m2.df_model - m1.df_model
    f_val = ((m1.ssr - m2.ssr) / df_diff) / (m2.ssr / m2.df_resid)
    p_val = sp.f.sf(f_val, df_diff, m2.df_resid)
    return m2.rsquared - m1.rsquared, f_val, p_val, m1, m2


def load_data():
    """Primary scorer: Qwen3-ORPO-400 logit-EV (ic_q400) indexed to the
    Salvi texts_for_scoring.jsonl row order. Cross-scorer: gpt-4.1-mini on
    the same pre-treatment text (IC_openai_initial in all_complexity_scores.csv)."""
    an = pd.read_csv(DATA / "analysis_data.csv")

    # Primary Q400 logit-EV
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q400 = pd.read_csv(DATA_Q400 /
                       "salvi_texts_for_scoring_initial_qwenorpo400.csv")
    assert len(meta) == len(q400)
    ic_q400 = pd.DataFrame({
        "participantId": [m.get("participantId") for m in meta],
        "ic_q400": q400["ic_qwenorpo400_logit"].astype(float).values,
    })

    # Cross-scorer (gpt-4.1-mini on pre-treatment text)
    ic_old = pd.read_csv(DATA / "all_complexity_scores.csv")
    ic_col_name = "IC_openai_initial" if "IC_openai_initial" in ic_old.columns else "IC_openai"
    ic_old = ic_old.rename(columns={ic_col_name: "ic_openai_pre"})

    df = an.merge(ic_old, on="participantId", how="inner")
    df = df.merge(ic_q400, on="participantId", how="left")

    for c in ["opinion_change", "abs_opinion_change", "is_ai_opponent",
              "wc_all", "wc_initial", "age", "is_male",
              "agreementPreTreatment"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ── Analysis 1: Quadratic Moderation (Full Model) ─────────────────────────

def analysis_full_moderation(df, ic_col, label):
    """DV ~ IC + IC² + pre-agreement + wc_initial (paper's canonical spec for Salvi)."""
    print(f"\n{'='*70}")
    print(f"  FULL MODERATION — {label}")
    print(f"{'='*70}")

    covs = [c for c in ["agreementPreTreatment", "wc_initial"] if c in df.columns]

    results = {}
    for dv, dv_label in [("opinion_change", "Opinion change (signed)"),
                         ("abs_opinion_change", "Absolute opinion change")]:
        if dv not in df.columns:
            continue

        sub = df.dropna(subset=[dv, ic_col] + covs).copy()
        y = sub[dv].values

        ic = zs(sub[ic_col].values)
        ic_sq = zs(sub[ic_col].values ** 2)
        cov_stack = [zs(sub[c].values) for c in covs]

        X_lin = sm.add_constant(np.column_stack([ic] + cov_stack))
        X_quad = sm.add_constant(np.column_stack([ic, ic_sq] + cov_stack))
        dr2, f_val, p_val, m_lin, m_quad = delta_r2_test(y, X_lin, X_quad)

        print(f"\n  {dv_label}: n = {len(sub)} (covs: {covs})")
        print(f"    β_IC = {m_quad.params[1]:+.3f}, p = {fmt_p(m_quad.pvalues[1])}")
        print(f"    β_IC² = {m_quad.params[2]:+.3f}, p = {fmt_p(m_quad.pvalues[2])}")
        print(f"    R² = {m_quad.rsquared:.4f}, ΔR² = {dr2:.4f}, p = {fmt_p(p_val)}")

        results[dv] = {
            "n": len(sub),
            "beta_ic2": m_quad.params[2],
            "p_ic2": m_quad.pvalues[2],
            "dr2": dr2,
            "p": p_val,
        }

    return results


# ── Analysis 2: Condition-Specific Models ─────────────────────────────────

def analysis_condition_specific(df, ic_col, label):
    """Separate models for AI vs human opponent (with canonical covariates)."""
    print(f"\n{'='*70}")
    print(f"  CONDITION-SPECIFIC — {label}")
    print(f"{'='*70}")

    dv = "opinion_change"
    covs = [c for c in ["agreementPreTreatment", "wc_initial"] if c in df.columns]
    results = {}

    for cond_val, cond_name in [(1, "AI opponent"), (0, "Human opponent")]:
        sub = df[df["is_ai_opponent"] == cond_val].dropna(subset=[dv, ic_col] + covs).copy()
        if len(sub) < 30:
            print(f"\n  {cond_name}: n = {len(sub)} (too few)")
            continue

        y = sub[dv].values
        ic = zs(sub[ic_col].values)
        ic_sq = zs(sub[ic_col].values ** 2)
        cov_stack = [zs(sub[c].values) for c in covs]

        X = sm.add_constant(np.column_stack([ic, ic_sq] + cov_stack))
        m = sm.OLS(y, X).fit()

        print(f"\n  {cond_name}: n = {len(sub)}")
        print(f"    β_IC = {m.params[1]:+.3f}, p = {fmt_p(m.pvalues[1])}")
        print(f"    β_IC² = {m.params[2]:+.3f}, p = {fmt_p(m.pvalues[2])}")
        print(f"    R² = {m.rsquared:.4f}")

        results[cond_name] = {
            "n": len(sub),
            "beta_ic2": m.params[2],
            "p_ic2": m.pvalues[2],
        }

    return results


# ── Analysis 3: IC × Condition Interaction ────────────────────────────────

def analysis_interaction(df, ic_col, label):
    """Full interaction model: DV ~ IC + IC² + cond + IC×cond + IC²×cond."""
    print(f"\n{'='*70}")
    print(f"  INTERACTION MODEL — {label}")
    print(f"{'='*70}")

    dv = "opinion_change"
    covs = [c for c in ["agreementPreTreatment", "wc_initial"] if c in df.columns]
    sub = df.dropna(subset=[dv, ic_col, "is_ai_opponent"] + covs).copy()
    y = sub[dv].values

    ic = zs(sub[ic_col].values)
    ic_sq = zs(sub[ic_col].values ** 2)
    cond = sub["is_ai_opponent"].values.astype(float)
    cov_stack = [zs(sub[c].values) for c in covs]

    X = sm.add_constant(np.column_stack([
        ic, ic_sq, cond, ic * cond, ic_sq * cond
    ] + cov_stack))
    m = sm.OLS(y, X).fit()

    names = ["const", ic_col, f"{ic_col}²", "is_ai",
             f"{ic_col}×ai", f"{ic_col}²×ai"] + covs
    print(f"\n  n = {len(sub)}, R² = {m.rsquared:.4f}")
    print(f"\n  {'Term':<25s} {'β':>8s} {'SE':>8s} {'p':>8s}")
    print(f"  {'-'*55}")
    for i, name in enumerate(names):
        print(f"  {name:<25s} {m.params[i]:>+8.3f} {m.bse[i]:>8.3f} {fmt_p(m.pvalues[i]):>8s}")


# ── Analysis 4: Absolute Change (Engagement Intensity) ────────────────────

def analysis_absolute_change(df, ic_col, label):
    """IC predicts magnitude of opinion shift regardless of direction."""
    print(f"\n{'='*70}")
    print(f"  ABSOLUTE CHANGE — {label}")
    print(f"{'='*70}")

    dv = "abs_opinion_change"
    if dv not in df.columns:
        print("  (abs_opinion_change not available)")
        return

    covs = [c for c in ["agreementPreTreatment", "wc_initial"] if c in df.columns]
    sub = df.dropna(subset=[dv, ic_col] + covs).copy()
    y = sub[dv].values

    ic = zs(sub[ic_col].values)
    ic_sq = zs(sub[ic_col].values ** 2)
    cov_stack = [zs(sub[c].values) for c in covs]

    X_lin = sm.add_constant(np.column_stack([ic] + cov_stack))
    X_quad = sm.add_constant(np.column_stack([ic, ic_sq] + cov_stack))
    dr2, f_val, p_val, _, m_quad = delta_r2_test(y, X_lin, X_quad)

    print(f"\n  n = {len(sub)}")
    print(f"  β_IC = {m_quad.params[1]:+.3f}, p = {fmt_p(m_quad.pvalues[1])}")
    print(f"  β_IC² = {m_quad.params[2]:+.3f}, p = {fmt_p(m_quad.pvalues[2])}")
    print(f"  R² = {m_quad.rsquared:.4f}, ΔR² = {dr2:.4f}, p = {fmt_p(p_val)}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    print(f"Loaded {len(df)} participants")

    sub = df.dropna(subset=["ic_q400", "ic_openai_pre"])
    r, p = sp.pearsonr(sub["ic_q400"], sub["ic_openai_pre"])
    print(f"\nic_q400 vs ic_openai_pre: r = {r:.3f}, p = {fmt_p(p)}, n = {len(sub)}")
    print(f"ic_q400        mean = {sub['ic_q400'].mean():.2f}, SD = {sub['ic_q400'].std():.2f}")
    print(f"ic_openai_pre  mean = {sub['ic_openai_pre'].mean():.2f}, SD = {sub['ic_openai_pre'].std():.2f}")

    results = {}
    for ic_col, label in [("ic_q400", "Primary (Q400 logit-EV)"),
                          ("ic_openai_pre", "Cross-scorer (gpt-4.1-mini)")]:
        results[ic_col] = {}
        results[ic_col]["full"] = analysis_full_moderation(df, ic_col, label)
        results[ic_col]["cond"] = analysis_condition_specific(df, ic_col, label)
        analysis_interaction(df, ic_col, label)
        analysis_absolute_change(df, ic_col, label)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY: SALVI")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<40s} {'Q400':>12s} {'gpt-4.1-mini':>14s}")
    print(f"  {'-'*67}")
    for dv in ["opinion_change", "abs_opinion_change"]:
        if dv in results["ic_q400"]["full"]:
            for key, label in [("beta_ic2", f"  {dv} β_IC²"),
                               ("p_ic2", f"  {dv} p(β_IC²)")]:
                v1 = results["ic_q400"]["full"][dv][key]
                v2 = results["ic_openai_pre"]["full"][dv][key]
                if key.startswith("p"):
                    print(f"  {label:<40s} {fmt_p(v1):>12s} {fmt_p(v2):>14s}")
                else:
                    print(f"  {label:<40s} {v1:>12.3f} {v2:>14.3f}")


if __name__ == "__main__":
    main()
