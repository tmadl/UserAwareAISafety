#!/usr/bin/env python3
"""
06_absolute_change_engagement.py — IC and absolute outcome magnitude.

Tests whether IC predicts the absolute magnitude of belief/opinion change
(stripping direction) across datasets. Addresses the reviewer concern that
IC might capture "broader engagement or intensity" rather than directional
moderation of persuasion.

Key results:
  1. Costello treatment: IC → |belief change| (should be significant)
  2. Costello control:   IC → |belief change| (predicted null)
  3. IC × treatment interaction for |belief change|
  4. Salvi:              IC → |opinion shift| (significant — reported in paper)
  5. Cheng S3:           IC → within-condition |residual|
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
DATA_COSTELLO = ROOT / "data" / "costello2024"
DATA_CHENG = ROOT / "data" / "cheng2006"
DATA_SALVI = ROOT / "data" / "salvi2005"
DATA_Q400 = ROOT / "data" / "ic_qwen3orpo400"


# ── Utilities ──────────────────────────────────────────────────────────────

def zs(x):
    """Standardise (mean-centre, divide by SD)."""
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def delta_r2_test(y, X_reduced, X_full):
    """F-test for ΔR² between nested OLS models."""
    m1 = sm.OLS(y, X_reduced).fit()
    m2 = sm.OLS(y, X_full).fit()
    df_diff = m2.df_model - m1.df_model
    f_val = ((m1.ssr - m2.ssr) / df_diff) / (m2.ssr / m2.df_resid)
    p_val = sp.f.sf(f_val, df_diff, m2.df_resid)
    return m2.rsquared - m1.rsquared, f_val, p_val, m1, m2


# ── Data Loading ───────────────────────────────────────────────────────────

def load_costello():
    """Load Costello treatment + control with IC scores and belief change.

    Cross-arm IC measures, both scored on the pre-treatment text window:
      * ic_q400      — Qwen3-ORPO-400 logit-EV (paper's primary scorer).
                       Treatment: scored on text_initial via texts_for_scoring.jsonl;
                       control:   scored on conRestatement (costello_controls_qwenorpo400.csv).
      * ic_openai_pre — gpt-4.1-mini cross-scorer on the same pre-treatment text
                        (IC_openai_initial treatment; IC_openai control-arm file
                        scored on conRestatement).
    """
    # Treatment arm — analysis_data.csv (same filter as primary 01 analysis)
    an = pd.read_csv(DATA_COSTELLO / "analysis_data.csv")
    from _raw_data_check import require_raw
    _raw = DATA_COSTELLO / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
    require_raw(_raw, "Costello", "https://osf.io/gdkb7/")
    orig = pd.read_csv(_raw, low_memory=False)
    orig = orig.drop_duplicates(subset="participantId", keep="first")

    # Control arm — AllData rows with ExperimentalCondition == Control
    ctrl = orig[orig["ExperimentalCondition"] == "Control"].copy()

    # Primary Q400 logit-EV — treatment (jsonl-indexed to pre-treatment text)
    meta = [json.loads(l) for l in open(DATA_COSTELLO / "texts_for_scoring.jsonl")]
    q400_t = pd.read_csv(DATA_Q400 /
                         "costello_texts_for_scoring_initial_qwenorpo400.csv")
    assert len(meta) == len(q400_t)
    ic_q400_t = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "ic_q400": q400_t["ic_qwenorpo400_logit"].astype(float).values,
    })
    ic_q400_c = pd.read_csv(DATA_COSTELLO / "costello_controls_qwenorpo400.csv")[
        ["participantId", "ic_qwenorpo400_logit"]
    ].rename(columns={"ic_qwenorpo400_logit": "ic_q400"})

    # gpt-4.1-mini cross-scorer on pre-treatment text
    ic_openai_t = pd.read_csv(DATA_COSTELLO / "all_complexity_scores.csv")[
        ["participantId", "IC_openai_initial"]
    ].rename(columns={"IC_openai_initial": "ic_openai_pre"})
    ic_openai_c = pd.read_csv(DATA_COSTELLO / "control_ic_scores.csv")[
        ["participantId", "IC_openai"]
    ].rename(columns={"IC_openai": "ic_openai_pre"})

    # Treatment table (analysis_data subset; covariates from orig where missing)
    t = an.merge(ic_q400_t, on="participantId", how="inner")
    t = t.merge(ic_openai_t, on="participantId", how="left")
    for c in ["OpenendedResponseWordCount"]:
        if c not in t.columns:
            t = t.merge(orig[["participantId", c]], on="participantId", how="left")
    t["is_treatment"] = 1

    # Control table
    c = ctrl.merge(ic_q400_c, on="participantId", how="inner")
    c = c.merge(ic_openai_c, on="participantId", how="left")
    c["is_treatment"] = 0

    keep = ["participantId", "is_treatment", "DV_BeliefChange_Specific",
            "Pre_Belief_Specific", "OpenendedResponseWordCount",
            "ic_q400", "ic_openai_pre"]
    df = pd.concat([t[keep], c[keep]], ignore_index=True)

    for col in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
                "OpenendedResponseWordCount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["abs_change"] = df["DV_BeliefChange_Specific"].abs()

    return df


def load_salvi():
    """Load Salvi analysis data with IC scores."""
    an = pd.read_csv(DATA_SALVI / "analysis_data.csv")
    ic = pd.read_csv(DATA_SALVI / "all_complexity_scores.csv")
    df = an.merge(ic, on="participantId", how="inner")
    for c in ["opinion_change", "abs_opinion_change"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_cheng():
    """Load Cheng analysis data with IC scores (Study 3)."""
    an = pd.read_csv(DATA_CHENG / "analysis_data.csv")
    ic = pd.read_csv(DATA_CHENG / "all_complexity_scores.csv")
    df = an.merge(ic, on="participantId", how="inner")
    for c in ["rightorwrong", "is_sycophantic"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["study"] == 3].copy()


# ── Analysis 1: Costello — IC → |belief change| by condition ─────────────

def analysis_costello_absolute(df, ic_col="ic_q400", label="Primary (Q400 logit-EV)"):
    """DV = |Δ|. Canonical spec: z(IC) + z(IC²) + pre-belief + word count.
    Reports both parameterizations for the control arm so the paper's
    placebo comparator numbers are recoverable in either convention."""
    print(f"\n{'='*70}")
    print(f"  COSTELLO: IC → |BELIEF CHANGE| BY CONDITION — {label}")
    print(f"{'='*70}")

    covs = ["Pre_Belief_Specific", "OpenendedResponseWordCount"]
    need = [ic_col, "abs_change"] + covs
    df = df.dropna(subset=need).copy()
    print(f"\n  Total N = {len(df)} "
          f"(treatment: {df.is_treatment.sum()}, "
          f"control: {(1 - df.is_treatment).sum()})")
    print(f"  mean |Δ| treat = {df.loc[df.is_treatment == 1, 'abs_change'].mean():.2f}, "
          f"control = {df.loc[df.is_treatment == 0, 'abs_change'].mean():.2f} "
          f"(ratio {df.loc[df.is_treatment == 1, 'abs_change'].mean() / df.loc[df.is_treatment == 0, 'abs_change'].mean():.2f}x)")

    results = {}
    for grp, arm_label in [(1, "Treatment"), (0, "Control")]:
        sub = df[df["is_treatment"] == grp].copy()
        y = sub["abs_change"].values

        ic_raw = sub[ic_col].values
        ic = zs(ic_raw)
        ic_sq_z = zs(ic_raw ** 2)
        pre = zs(sub[covs[0]].values)
        wc = zs(sub[covs[1]].values)

        # Linear-only (for linear p-value and ΔR²)
        X_lin = sm.add_constant(np.column_stack([ic, pre, wc]))
        # Full quadratic (canonical: z(IC) + z(IC²) + covs)
        X_full = sm.add_constant(np.column_stack([ic, ic_sq_z, pre, wc]))
        dr2, f_val, p_val, m_lin, m_full = delta_r2_test(y, X_lin, X_full)

        # Raw-IC² alternate parameterization (rawIC + rawIC² + rawcovs)
        X_raw = sm.add_constant(np.column_stack([
            ic_raw, ic_raw ** 2, sub[covs[0]].values, sub[covs[1]].values]))
        m_raw = sm.OLS(y, X_raw).fit()

        # |pre - 50| sensitivity
        abs_pre_z = zs((sub[covs[0]].values - 50).__abs__())
        X_abspre = sm.add_constant(np.column_stack(
            [ic, ic_sq_z, pre, wc, abs_pre_z]))
        m_abspre = sm.OLS(y, X_abspre).fit()

        print(f"\n  {arm_label} (N = {len(sub)}):")
        print(f"    Linear-only β_IC = {m_lin.params[1]:+.3f}, p = {fmt_p(m_lin.pvalues[1])}")
        print(f"    Quadratic ΔR² = {dr2:.4f}, F(1, {int(m_full.df_resid)}) = {f_val:.2f}, p = {fmt_p(p_val)}")
        print(f"    z(IC) + z(IC²):  β_IC = {m_full.params[1]:+.3f}, "
              f"β_z(IC²) = {m_full.params[2]:+.3f}, p = {fmt_p(m_full.pvalues[2])}")
        print(f"    raw IC + rawIC²: β_IC = {m_raw.params[1]:+.3f}, "
              f"β_rawIC² = {m_raw.params[2]:+.3f}, p = {fmt_p(m_raw.pvalues[2])}")
        print(f"    + |pre − 50|:    β_z(IC²) = {m_abspre.params[2]:+.3f}, "
              f"p = {fmt_p(m_abspre.pvalues[2])}")

        results[arm_label] = {
            "n": len(sub), "dr2": dr2, "f": f_val, "p": p_val,
            "beta_ic_linear": m_lin.params[1], "p_ic_linear": m_lin.pvalues[1],
            "beta_zicsq": m_full.params[2], "p_zicsq": m_full.pvalues[2],
            "beta_rawicsq": m_raw.params[2], "p_rawicsq": m_raw.pvalues[2],
            "beta_zicsq_abspre": m_abspre.params[2],
        }

    return results


# ── Analysis 2: Costello — IC × treatment interaction for |change| ───────

def analysis_costello_interaction(df, ic_col="ic_q400", label="Primary (Q400 logit-EV)"):
    print(f"\n{'='*70}")
    print(f"  COSTELLO: IC × TREATMENT INTERACTION FOR |BELIEF CHANGE| — {label}")
    print(f"{'='*70}")

    covs = ["Pre_Belief_Specific", "OpenendedResponseWordCount"]
    df = df.dropna(subset=[ic_col, "abs_change"] + covs).copy()

    ic_raw = df[ic_col].values
    ic = zs(ic_raw)
    ic_sq = zs(ic_raw ** 2)
    treat = df["is_treatment"].values.astype(float)
    pre = zs(df[covs[0]].values)
    wc = zs(df[covs[1]].values)

    y = df["abs_change"].values

    # Base model: IC + IC² + treatment + covs
    X_base = sm.add_constant(np.column_stack([ic, ic_sq, treat, pre, wc]))
    # Full model: + IC×treatment + IC²×treatment
    X_full = sm.add_constant(np.column_stack([
        ic, ic_sq, treat, ic * treat, ic_sq * treat, pre, wc
    ]))

    dr2, f_val, p_val, m_base, m_full = delta_r2_test(y, X_base, X_full)

    print(f"\n  N = {len(df)}")
    print(f"  Base R² = {m_base.rsquared:.4f}")
    print(f"  Full R² = {m_full.rsquared:.4f}")
    print(f"  ΔR²     = {dr2:.4f}, F(2, {int(m_full.df_resid)}) = {f_val:.2f}, "
          f"p = {fmt_p(p_val)}")

    names = ["const", "IC", "IC²", "treatment",
             "IC×treatment", "IC²×treatment", "pre_belief", "word_count"]
    print(f"\n  {'Term':<25s} {'β':>8s} {'SE':>8s} {'p':>8s}")
    print(f"  {'-'*55}")
    for i, name in enumerate(names):
        print(f"  {name:<25s} {m_full.params[i]:>+8.3f} "
              f"{m_full.bse[i]:>8.3f} {fmt_p(m_full.pvalues[i]):>8s}")

    return {
        "n": len(df), "dr2": dr2, "f": f_val, "p": p_val,
    }


# ── Analysis 2b: Costello — Treatment × IC interaction for SIGNED DV ─────

def analysis_costello_signed_interaction(df, ic_col="IC_openai", label="IC_openai"):
    """Reviewer Overall #2: pooled-arm test that the quadratic IC effect on
    signed belief change is treatment-specific.

    DV_signed ~ IC + IC² + Treatment + IC×Treatment + IC²×Treatment + covariates.
    A negative, significant IC²×Treatment indicates the inverted-U is
    concentrated in the AI-dialogue arm (absent in controls), ruling out that
    IC simply tracks engagement/intensity.
    """
    print(f"\n{'='*70}")
    print(f"  COSTELLO: TREATMENT × IC INTERACTION FOR SIGNED BELIEF CHANGE — {label}")
    print(f"{'='*70}")

    covs = ["Pre_Belief_Specific", "OpenendedResponseWordCount"]
    need = [ic_col, "DV_BeliefChange_Specific", "is_treatment"] + covs
    sub = df.dropna(subset=need).copy()

    y = sub["DV_BeliefChange_Specific"].values
    ic = zs(sub[ic_col].values)
    ic_sq = zs(sub[ic_col].values ** 2)
    treat = sub["is_treatment"].values.astype(float)
    pre = zs(sub[covs[0]].values)
    wc = zs(sub[covs[1]].values)

    # Base: IC + IC² + Treatment + covs (additive only)
    X_base = sm.add_constant(np.column_stack([ic, ic_sq, treat, pre, wc]))
    # +IC×Treatment
    X_lin_ix = sm.add_constant(np.column_stack([
        ic, ic_sq, treat, ic * treat, pre, wc]))
    # +IC×Treatment +IC²×Treatment  (full)
    X_full = sm.add_constant(np.column_stack([
        ic, ic_sq, treat, ic * treat, ic_sq * treat, pre, wc]))

    # Nested test: base vs full (both interactions jointly)
    dr2_joint, f_joint, p_joint, m_base, m_full = delta_r2_test(y, X_base, X_full)
    # Isolated test of IC²×Treatment (controlling for linear interaction)
    dr2_sq, f_sq, p_sq, _, _ = delta_r2_test(y, X_lin_ix, X_full)

    print(f"\n  N = {len(sub)} "
          f"(treatment: {int(treat.sum())}, control: {int((1 - treat).sum())})")
    print(f"\n  Base (additive IC + IC² + Treatment + covs):  R² = {m_base.rsquared:.4f}")
    print(f"  Full (+ IC×Treat + IC²×Treat):                R² = {m_full.rsquared:.4f}")
    print(f"\n  Joint interaction test   ΔR² = {dr2_joint:.4f}, "
          f"F(2, {int(m_full.df_resid)}) = {f_joint:.2f}, p = {fmt_p(p_joint)}")
    print(f"  IC²×Treat (isolated)     ΔR² = {dr2_sq:.4f}, "
          f"F(1, {int(m_full.df_resid)}) = {f_sq:.2f}, p = {fmt_p(p_sq)}")

    names = ["const", "IC", "IC²", "Treatment",
             "IC×Treat", "IC²×Treat", "pre_belief", "word_count"]
    print(f"\n  {'Term':<20s} {'β':>8s} {'SE':>8s} {'p':>8s}")
    print(f"  {'-'*50}")
    for i, name in enumerate(names):
        print(f"  {name:<20s} {m_full.params[i]:>+8.3f} "
              f"{m_full.bse[i]:>8.3f} {fmt_p(m_full.pvalues[i]):>8s}")

    # Control-arm-only: verify quadratic IC is null among controls
    ctrl = sub[sub["is_treatment"] == 0].copy()
    y_c = ctrl["DV_BeliefChange_Specific"].values
    ic_c_raw = ctrl[ic_col].values
    ic_c = zs(ic_c_raw); ic_sq_c = zs(ic_c_raw ** 2)
    pre_c = zs(ctrl[covs[0]].values); wc_c = zs(ctrl[covs[1]].values)
    X0_c = sm.add_constant(np.column_stack([ic_c, pre_c, wc_c]))
    X1_c = sm.add_constant(np.column_stack([ic_c, ic_sq_c, pre_c, wc_c]))
    dr2_c, f_c, p_c, _, m_c = delta_r2_test(y_c, X0_c, X1_c)
    # Raw-IC² alternate parameterization (for paper's placebo comparator)
    X_c_raw = sm.add_constant(np.column_stack(
        [ic_c_raw, ic_c_raw ** 2, ctrl[covs[0]].values, ctrl[covs[1]].values]))
    m_c_raw = sm.OLS(y_c, X_c_raw).fit()
    print(f"\n  Control arm only (N = {len(ctrl)}): "
          f"β_z(IC²) = {m_c.params[2]:+.3f} (p = {fmt_p(m_c.pvalues[2])}), "
          f"β_rawIC² = {m_c_raw.params[2]:+.3f}")
    print(f"    linear β_IC = {m_c.params[1]:+.3f} (p = {fmt_p(m_c.pvalues[1])})")
    print(f"    Quadratic ΔR² vs linear: {dr2_c:.4f}, p = {fmt_p(p_c)}")

    # Treatment-arm-only: for symmetry, report the quadratic
    tr = sub[sub["is_treatment"] == 1].copy()
    y_t = tr["DV_BeliefChange_Specific"].values
    ic_t_raw = tr[ic_col].values
    ic_t = zs(ic_t_raw); ic_sq_t = zs(ic_t_raw ** 2)
    pre_t = zs(tr[covs[0]].values); wc_t = zs(tr[covs[1]].values)
    X0_t = sm.add_constant(np.column_stack([ic_t, pre_t, wc_t]))
    X1_t = sm.add_constant(np.column_stack([ic_t, ic_sq_t, pre_t, wc_t]))
    dr2_t, f_t, p_t, _, m_t = delta_r2_test(y_t, X0_t, X1_t)
    X_t_raw = sm.add_constant(np.column_stack(
        [ic_t_raw, ic_t_raw ** 2, tr[covs[0]].values, tr[covs[1]].values]))
    m_t_raw = sm.OLS(y_t, X_t_raw).fit()
    print(f"  Treatment arm only (N = {len(tr)}): "
          f"β_z(IC²) = {m_t.params[2]:+.3f} (p = {fmt_p(m_t.pvalues[2])}), "
          f"β_rawIC² = {m_t_raw.params[2]:+.3f}")
    print(f"    linear β_IC = {m_t.params[1]:+.3f} (p = {fmt_p(m_t.pvalues[1])})")
    print(f"    Quadratic ΔR² vs linear: {dr2_t:.4f}, p = {fmt_p(p_t)}")

    return {
        "n": len(sub), "n_treat": int(treat.sum()), "n_ctrl": int((1 - treat).sum()),
        "dr2_joint": dr2_joint, "f_joint": f_joint, "p_joint": p_joint,
        "dr2_sq": dr2_sq, "f_sq": f_sq, "p_sq": p_sq,
        "beta_ic_x_treat": m_full.params[4], "p_ic_x_treat": m_full.pvalues[4],
        "beta_icsq_x_treat": m_full.params[5], "p_icsq_x_treat": m_full.pvalues[5],
        "ctrl": {"n": len(ctrl), "beta_ic": m_c.params[1], "p_ic": m_c.pvalues[1],
                 "beta_icsq": m_c.params[2], "p_icsq": m_c.pvalues[2],
                 "dr2": dr2_c, "p": p_c},
        "treat": {"n": len(tr), "beta_ic": m_t.params[1], "p_ic": m_t.pvalues[1],
                  "beta_icsq": m_t.params[2], "p_icsq": m_t.pvalues[2],
                  "dr2": dr2_t, "p": p_t},
    }


# ── Analysis 3: Salvi — IC → |opinion shift| ────────────────────────────

def analysis_salvi_absolute(df):
    print(f"\n{'='*70}")
    print(f"  SALVI: IC → |OPINION SHIFT|")
    print(f"{'='*70}")

    dv = "abs_opinion_change"
    sub = df.dropna(subset=[dv, "IC_openai"]).copy()
    y = sub[dv].values

    ic = zs(sub["IC_openai"].values)
    ic_sq = ic ** 2

    X0 = sm.add_constant(np.ones(len(sub)))
    X1 = sm.add_constant(np.column_stack([ic, ic_sq]))
    dr2, f_val, p_val, _, m1 = delta_r2_test(y, X0, X1)

    print(f"\n  N = {len(sub)}")
    print(f"  ΔR² = {dr2:.4f}, F(2, {int(m1.df_resid)}) = {f_val:.2f}, "
          f"p = {fmt_p(p_val)}")
    print(f"  β_IC  = {m1.params[1]:+.3f}, p = {fmt_p(m1.pvalues[1])}")
    print(f"  β_IC² = {m1.params[2]:+.3f}, p = {fmt_p(m1.pvalues[2])}")

    return {"n": len(sub), "dr2": dr2, "f": f_val, "p": p_val}


# ── Analysis 4: Cheng S3 — IC → within-condition |residual| ─────────────

def analysis_cheng_absolute(df):
    print(f"\n{'='*70}")
    print(f"  CHENG S3: IC → WITHIN-CONDITION |RESIDUAL|")
    print(f"{'='*70}")

    sub = df.dropna(subset=["IC_openai", "rightorwrong", "is_sycophantic"]).copy()
    print(f"\n  N = {len(sub)}")

    ic = zs(sub["IC_openai"].values)
    ic_sq = ic ** 2

    # Within-condition absolute residual (strips condition-level direction)
    cond_means = sub.groupby("is_sycophantic")["rightorwrong"].transform("mean")
    sub["abs_resid"] = (sub["rightorwrong"] - cond_means).abs()

    y = sub["abs_resid"].values
    X0 = sm.add_constant(np.ones(len(sub)))
    X1 = sm.add_constant(np.column_stack([ic, ic_sq]))
    dr2, f_val, p_val, _, m1 = delta_r2_test(y, X0, X1)

    print(f"\n  Within-condition |residual|:")
    print(f"    ΔR² = {dr2:.4f}, F(2, {int(m1.df_resid)}) = {f_val:.2f}, "
          f"p = {fmt_p(p_val)}")
    print(f"    β_IC  = {m1.params[1]:+.3f}, p = {fmt_p(m1.pvalues[1])}")
    print(f"    β_IC² = {m1.params[2]:+.3f}, p = {fmt_p(m1.pvalues[2])}")

    # Also test |DV - scale midpoint| for completeness
    sub["abs_midpoint"] = (sub["rightorwrong"] - 4).abs()
    y2 = sub["abs_midpoint"].values
    X0b = sm.add_constant(np.ones(len(sub)))
    X1b = sm.add_constant(np.column_stack([ic, ic_sq]))
    dr2b, f_valb, p_valb, _, m1b = delta_r2_test(y2, X0b, X1b)

    print(f"\n  |DV − scale midpoint| (exploratory):")
    print(f"    ΔR² = {dr2b:.4f}, F(2, {int(m1b.df_resid)}) = {f_valb:.2f}, "
          f"p = {fmt_p(p_valb)}")
    print(f"    β_IC  = {m1b.params[1]:+.3f}, p = {fmt_p(m1b.pvalues[1])}")
    print(f"    β_IC² = {m1b.params[2]:+.3f}, p = {fmt_p(m1b.pvalues[2])}")

    return {"n": len(sub), "dr2": dr2, "f": f_val, "p": p_val}


# ── Summary Table ────────────────────────────────────────────────────────

def print_summary(costello, interaction, salvi, cheng):
    print(f"\n{'='*70}")
    print(f"  SUMMARY: IC → |OUTCOME MAGNITUDE| ACROSS DATASETS")
    print(f"{'='*70}")

    print(f"\n  {'Dataset':<25s} {'N':>5s} {'ΔR²':>7s} {'F':>7s} {'p':>8s}")
    print(f"  {'-'*55}")
    print(f"  {'Costello (treatment)':<25s} {costello['Treatment']['n']:>5d} "
          f"{costello['Treatment']['dr2']:>7.4f} "
          f"{costello['Treatment']['f']:>7.2f} "
          f"{fmt_p(costello['Treatment']['p']):>8s}")
    print(f"  {'Costello (control)':<25s} {costello['Control']['n']:>5d} "
          f"{costello['Control']['dr2']:>7.4f} "
          f"{costello['Control']['f']:>7.2f} "
          f"{fmt_p(costello['Control']['p']):>8s}")
    print(f"  {'Salvi':<25s} {salvi['n']:>5d} "
          f"{salvi['dr2']:>7.4f} "
          f"{salvi['f']:>7.2f} "
          f"{fmt_p(salvi['p']):>8s}")
    print(f"  {'Cheng S3 (w/in cond)':<25s} {cheng['n']:>5d} "
          f"{cheng['dr2']:>7.4f} "
          f"{cheng['f']:>7.2f} "
          f"{fmt_p(cheng['p']):>8s}")

    print(f"\n  IC × treatment interaction (Costello |belief change|):")
    print(f"    N = {interaction['n']}, ΔR² = {interaction['dr2']:.4f}, "
          f"F(2, {interaction['n'] - 6}) = {interaction['f']:.2f}, "
          f"p = {fmt_p(interaction['p'])}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df_costello = load_costello()
    df_salvi = load_salvi()
    df_cheng = load_cheng()

    costello = analysis_costello_absolute(
        df_costello, ic_col="ic_q400", label="Primary (Q400 logit-EV)")
    costello_xs = analysis_costello_absolute(
        df_costello, ic_col="ic_openai_pre", label="Cross-scorer (gpt-4.1-mini)")
    interaction = analysis_costello_interaction(
        df_costello, ic_col="ic_q400", label="Primary (Q400 logit-EV)")
    signed_ix_orpo = analysis_costello_signed_interaction(
        df_costello, ic_col="ic_q400", label="Primary (Q400 logit-EV)")
    signed_ix = analysis_costello_signed_interaction(
        df_costello, ic_col="ic_openai_pre", label="Cross-scorer (gpt-4.1-mini)")
    salvi = analysis_salvi_absolute(df_salvi)
    cheng = analysis_cheng_absolute(df_cheng)

    print_summary(costello, interaction, salvi, cheng)

    print(f"\n{'='*70}")
    print(f"  SIGNED-DV TREATMENT × IC INTERACTION (Reviewer Overall #2)")
    print(f"{'='*70}")
    for tag, r in [("Q400 (primary)", signed_ix_orpo), ("gpt-4.1-mini (cross)", signed_ix)]:
        print(f"\n  [{tag}]")
        print(f"    Joint (IC×Treat + IC²×Treat): ΔR² = {r['dr2_joint']:.4f}, "
              f"F = {r['f_joint']:.2f}, p = {fmt_p(r['p_joint'])}")
        print(f"    IC²×Treat: β = {r['beta_icsq_x_treat']:+.3f}, p = {fmt_p(r['p_icsq_x_treat'])}")
        print(f"    IC ×Treat: β = {r['beta_ic_x_treat']:+.3f}, p = {fmt_p(r['p_ic_x_treat'])}")
        print(f"    Treat arm β_IC² = {r['treat']['beta_icsq']:+.3f} (p={fmt_p(r['treat']['p_icsq'])}), "
              f"Ctrl arm β_IC² = {r['ctrl']['beta_icsq']:+.3f} (p={fmt_p(r['ctrl']['p_icsq'])})")


if __name__ == "__main__":
    main()
