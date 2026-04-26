#!/usr/bin/env python3
"""
01_costello_analysis.py — Costello et al. reanalysis (evidence-based persuasion).

Reproduces the paper's core analyses for Costello data:
  1. Quadratic IC moderation of belief change (inverted-U)
  2. Within-study replication (Studies 1, 2, 3)
  3. Quintile analysis
  4. Discriminant validity (IC beyond word count + pre-belief)
  5. Competing moderators (24 variables)
  6. IC × argument quality interaction

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
DATA = ROOT / "data" / "costello2024"
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

def load_data():
    """Merge Costello analysis_data with primary (Q400 logit-EV) and
    cross-scorer (gpt-4.1-mini) IC scores.

    The paper's primary scorer is Qwen3-ORPO-400 with logit-expected-value
    decoding, which yields a continuous score on [1, 7] (released as
    `tmadl/IC-Qwen3.5-ORPO-400` on HuggingFace).
    """
    an = pd.read_csv(DATA / "analysis_data.csv")
    # all_complexity_scores.csv has IC_openai_initial (scored on text_initial,
    # same pre-treatment window as primary) and IC_openai (scored on the
    # concatenated conversation text). Cross-scorer for paper numbers is the
    # former; the latter is kept only as ancillary.
    ic_old = pd.read_csv(DATA / "all_complexity_scores.csv")
    ic_old = ic_old.rename(columns={"IC_openai_initial": "IC_openai_pre",
                                     "IC_openai_pre": "IC_openai_concat"})

    # Primary: Q400 logit-EV (row-indexed to texts_for_scoring.jsonl)
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q400 = pd.read_csv(DATA_Q400 /
                       "costello_texts_for_scoring_initial_qwenorpo400.csv")
    assert len(meta) == len(q400), "jsonl / q400 csv length mismatch"
    ic_q400 = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC_q400_logit": q400["ic_qwenorpo400_logit"].astype(float).values,
    })

    df = an.merge(ic_old, on="participantId", how="inner")
    df = df.merge(ic_q400, on="participantId", how="left")

    # Load original data for competing moderators
    orig_path = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
    if orig_path.exists():
        orig = pd.read_csv(orig_path, low_memory=False)
        pid_level = orig.drop_duplicates(subset=["participantId"], keep="first")
        want = ["participantId", "IH", "AOT", "Misinformation_Resis",
                "Social_Influence2", "GeneralTrust", "PersonalTrust",
                "genai_fam_1", "genai_use_1", "Sureness_1",
                "GPT_CoT_PlausibilityRating"]
        want = [c for c in want if c in pid_level.columns]
        extra = pid_level[want].copy()
        for c in extra.columns:
            if c != "participantId":
                extra[c] = pd.to_numeric(extra[c], errors="coerce")
        df = df.merge(extra, on="participantId", how="left")

    # Numerics
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "StudyNumber"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Encode categoricals
    if "Education_Cat" in df.columns:
        df["edu_num"] = df["Education_Cat"].map({
            "LessThanHighSchool": 1, "HighSchool": 2, "SomeCollege": 3,
            "Associate": 3, "Bachelors": 4, "Masters": 5, "JD/MD": 6, "PhD": 6})
    if "GenderCat" in df.columns:
        df["is_male"] = (df["GenderCat"] == "Male").astype(float)
    if "PartyCat" in df.columns:
        df["is_republican"] = df["PartyCat"].str.contains("Repub", na=False).astype(float)

    return df


# ── Analysis 1: Quadratic Moderation ──────────────────────────────────────

def analysis_quadratic_moderation(df, ic_col, label):
    """DV ~ IC + IC² + pre-belief + word count. Test ΔR² for IC² term."""
    print(f"\n{'='*70}")
    print(f"  QUADRATIC MODERATION — {label}")
    print(f"{'='*70}")

    dv = "DV_BeliefChange_Specific"
    covs = ["Pre_Belief_Specific", "OpenendedResponseWordCount"]

    sub = df.dropna(subset=[dv, ic_col] + covs).copy()
    y = sub[dv].values
    n = len(sub)

    ic = zs(sub[ic_col].values)
    ic_sq = zs(sub[ic_col].values ** 2)
    pre = zs(sub[covs[0]].values)
    wc = zs(sub[covs[1]].values)

    # Linear model: DV ~ IC + pre + wc
    X_lin = sm.add_constant(np.column_stack([ic, pre, wc]))
    # Quadratic model: DV ~ IC + IC² + pre + wc
    X_quad = sm.add_constant(np.column_stack([ic, ic_sq, pre, wc]))

    dr2, f_val, p_val, m_lin, m_quad = delta_r2_test(y, X_lin, X_quad)

    print(f"\n  n = {n}")
    print(f"  Linear R² = {m_lin.rsquared:.4f}")
    print(f"  Quadratic R² = {m_quad.rsquared:.4f}")
    print(f"  ΔR² = {dr2:.4f}, F(1,{int(m_quad.df_resid)}) = {f_val:.2f}, p = {fmt_p(p_val)}")

    # Coefficients
    names = ["const", ic_col, f"{ic_col}²", "pre_belief", "word_count"]
    print(f"\n  {'Term':<25s} {'β':>8s} {'SE':>8s} {'p':>8s}")
    print(f"  {'-'*55}")
    for i, name in enumerate(names):
        b = m_quad.params[i]
        se = m_quad.bse[i]
        p = m_quad.pvalues[i]
        print(f"  {name:<25s} {b:>+8.3f} {se:>8.3f} {fmt_p(p):>8s}")

    # β_IC (linear) in linear-only model
    b_lin = m_lin.params[1]
    p_lin = m_lin.pvalues[1]
    print(f"\n  Linear-only β_IC = {b_lin:+.3f}, p = {fmt_p(p_lin)}")

    return {
        "n": n,
        "dr2": dr2,
        "f": f_val,
        "p": p_val,
        "beta_ic2": m_quad.params[2],
        "p_ic2": m_quad.pvalues[2],
        "beta_ic_linear": b_lin,
        "p_ic_linear": p_lin,
        "r2_quad": m_quad.rsquared,
    }


# ── Analysis 2: Within-Study Replication ──────────────────────────────────

def analysis_within_study(df, ic_col, label):
    """Run quadratic model separately for Studies 1, 2, 3."""
    print(f"\n{'='*70}")
    print(f"  WITHIN-STUDY REPLICATION — {label}")
    print(f"{'='*70}")

    dv = "DV_BeliefChange_Specific"
    covs = ["Pre_Belief_Specific", "OpenendedResponseWordCount"]

    for study in [1, 2, 3]:
        sub = df[df["StudyNumber"] == study].dropna(subset=[dv, ic_col] + covs).copy()
        if len(sub) < 30:
            print(f"\n  Study {study}: n = {len(sub)} (too few)")
            continue

        y = sub[dv].values
        ic = zs(sub[ic_col].values)
        ic_sq = zs(sub[ic_col].values ** 2)
        pre = zs(sub[covs[0]].values)
        wc = zs(sub[covs[1]].values)

        X_lin = sm.add_constant(np.column_stack([ic, pre, wc]))
        X_quad = sm.add_constant(np.column_stack([ic, ic_sq, pre, wc]))
        dr2, f_val, p_val, _, m = delta_r2_test(y, X_lin, X_quad)

        print(f"\n  Study {study}: n = {len(sub)}, "
              f"β_IC² = {m.params[2]:+.3f}, p = {fmt_p(m.pvalues[2])}, "
              f"ΔR² = {dr2:.4f}, p = {fmt_p(p_val)}")


# ── Analysis 3: Quintile Analysis ─────────────────────────────────────────

def analysis_quintiles(df, ic_col, label):
    """IC quintile means for belief change with bootstrapped 95% CIs."""
    print(f"\n{'='*70}")
    print(f"  QUINTILE ANALYSIS — {label}")
    print(f"{'='*70}")

    dv = "DV_BeliefChange_Specific"
    sub = df.dropna(subset=[dv, ic_col]).copy()
    sub["ic_q"] = pd.qcut(sub[ic_col], 5, labels=False, duplicates="drop")

    print(f"\n  {'Q':<4s} {'n':>5s} {'IC mean':>8s} {'DV mean':>10s} {'DV 95% CI':>20s}")
    print(f"  {'-'*50}")
    for q in sorted(sub["ic_q"].unique()):
        s = sub[sub["ic_q"] == q]
        m = s[dv].mean()
        # Bootstrap CI
        rng = np.random.default_rng(42)
        boots = [s[dv].sample(len(s), replace=True, random_state=int(rng.integers(1e9))).mean()
                 for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  Q{q+1:<3d} {len(s):>5d} {s[ic_col].mean():>8.2f} {m:>10.1f} [{lo:>7.1f}, {hi:>7.1f}]")


# ── Analysis 4: Discriminant Validity ─────────────────────────────────────

def analysis_discriminant_validity(df, ic_col, label):
    """IC predicts belief change beyond word count, pre-belief, education."""
    print(f"\n{'='*70}")
    print(f"  DISCRIMINANT VALIDITY — {label}")
    print(f"{'='*70}")

    dv = "DV_BeliefChange_Specific"
    surface_covs = ["OpenendedResponseWordCount", "Pre_Belief_Specific"]

    sub = df.dropna(subset=[dv, ic_col] + surface_covs).copy()
    y = sub[dv].values

    X_covs = sm.add_constant(np.column_stack([zs(sub[c].values) for c in surface_covs]))
    X_full = sm.add_constant(np.column_stack(
        [zs(sub[c].values) for c in surface_covs] +
        [zs(sub[ic_col].values), zs(sub[ic_col].values ** 2)]
    ))

    dr2, f_val, p_val, m1, m2 = delta_r2_test(y, X_covs, X_full)
    print(f"\n  n = {len(sub)}")
    print(f"  Covariates R² = {m1.rsquared:.4f}")
    print(f"  + IC + IC²  R² = {m2.rsquared:.4f}")
    print(f"  ΔR² = {dr2:.4f}, F(2,{int(m2.df_resid)}) = {f_val:.2f}, p = {fmt_p(p_val)}")

    return {"dr2": dr2, "f": f_val, "p": p_val}


# ── Analysis 5: Competing Moderators ──────────────────────────────────────

def analysis_competing_moderators(df, ic_col, label):
    """Test each candidate moderator's quadratic β alongside IC."""
    print(f"\n{'='*70}")
    print(f"  COMPETING MODERATORS — {label}")
    print(f"{'='*70}")

    dv = "DV_BeliefChange_Specific"
    covs = ["Pre_Belief_Specific", "OpenendedResponseWordCount"]

    candidates = [
        ic_col, "IH", "AOT", "Pre_Belief_Specific",
        "edu_num", "AgeYears", "is_male", "is_republican",
        "OpenendedResponseWordCount",
    ]
    # Add personality/other cols if available
    for c in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness",
              "Neuroticism", "NeedForCognition", "CRT", "ParanoidIdeation",
              "ConspiracyMentality", "TrustInScience", "SocialMediaUse",
              "ReligiousAttendance"]:
        if c in df.columns:
            candidates.append(c)

    print(f"\n  {'Variable':<30s} {'n':>5s} {'β²':>8s} {'p(β²)':>8s} {'ΔR²':>8s} {'p(ΔR²)':>8s}")
    print(f"  {'-'*70}")

    results = []
    for var in candidates:
        if var not in df.columns:
            continue
        sub = df.dropna(subset=[dv, var] + covs).copy()
        if len(sub) < 50:
            continue

        y = sub[dv].values
        v = zs(sub[var].values)
        v_sq = zs(sub[var].values ** 2)
        pre = zs(sub[covs[0]].values)
        wc = zs(sub[covs[1]].values)

        X_lin = sm.add_constant(np.column_stack([v, pre, wc]))
        X_quad = sm.add_constant(np.column_stack([v, v_sq, pre, wc]))
        dr2, f_val, p_val, _, m_quad = delta_r2_test(y, X_lin, X_quad)

        sig = "*" if p_val < .05 else ""
        print(f"  {var:<30s} {len(sub):>5d} {m_quad.params[2]:>+8.3f} "
              f"{fmt_p(m_quad.pvalues[2]):>8s} {dr2:>8.4f} {fmt_p(p_val):>8s} {sig}")

        results.append({"var": var, "n": len(sub), "beta_sq": m_quad.params[2],
                        "p_sq": m_quad.pvalues[2], "dr2": dr2, "p_dr2": p_val})

    return results


# ── Analysis 6: IC × Argument Quality ─────────────────────────────────────

def analysis_argument_quality(df, ic_col, label):
    """IC × plausibility interaction on belief change."""
    print(f"\n{'='*70}")
    print(f"  IC × ARGUMENT QUALITY — {label}")
    print(f"{'='*70}")

    dv = "DV_BeliefChange_Specific"
    plaus_col = "GPT_CoT_PlausibilityRating"
    if plaus_col not in df.columns:
        print("  (plausibility ratings not available)")
        return

    covs = ["Pre_Belief_Specific", "OpenendedResponseWordCount"]
    sub = df.dropna(subset=[dv, ic_col, plaus_col] + covs).copy()
    y = sub[dv].values

    ic = zs(sub[ic_col].values)
    plaus = zs(sub[plaus_col].values)
    ix = ic * plaus
    pre = zs(sub[covs[0]].values)
    wc = zs(sub[covs[1]].values)

    X = sm.add_constant(np.column_stack([ic, plaus, ix, pre, wc]))
    m = sm.OLS(y, X).fit()

    names = ["const", ic_col, "plausibility", f"{ic_col}×plaus", "pre_belief", "word_count"]
    print(f"\n  n = {len(sub)}")
    print(f"  R² = {m.rsquared:.4f}")
    print(f"\n  {'Term':<30s} {'β':>8s} {'p':>8s}")
    print(f"  {'-'*48}")
    for i, name in enumerate(names):
        print(f"  {name:<30s} {m.params[i]:>+8.3f} {fmt_p(m.pvalues[i]):>8s}")

    # Bootstrap the interaction
    rng = np.random.default_rng(42)
    boot_betas = []
    for _ in range(10000):
        idx = rng.integers(0, len(y), size=len(y))
        try:
            mb = sm.OLS(y[idx], X[idx]).fit()
            boot_betas.append(mb.params[3])
        except Exception:
            pass
    lo, hi = np.percentile(boot_betas, [2.5, 97.5])
    print(f"\n  IC×plaus bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    print(f"Loaded {len(df)} participants")

    # Correlation between primary (Q400 logit-EV) and cross-scorer (gpt-4.1-mini)
    sub = df.dropna(subset=["IC_openai_pre", "IC_q400_logit"])
    r, p = sp.pearsonr(sub["IC_openai_pre"], sub["IC_q400_logit"])
    print(f"\nIC_openai vs IC_q400_logit: r = {r:.3f}, p = {fmt_p(p)}, n = {len(sub)}")
    print(f"IC_openai      mean = {sub['IC_openai'].mean():.2f}, SD = {sub['IC_openai'].std():.2f}")
    print(f"IC_q400_logit  mean = {sub['IC_q400_logit'].mean():.2f}, SD = {sub['IC_q400_logit'].std():.2f}")

    results = {}
    for ic_col, label in [("IC_q400_logit", "Primary (Q400 logit-EV)"),
                          ("IC_openai_pre", "Cross-scorer (gpt-4.1-mini)")]:
        results[ic_col] = {}
        results[ic_col]["quad"] = analysis_quadratic_moderation(df, ic_col, label)
        analysis_within_study(df, ic_col, label)
        analysis_quintiles(df, ic_col, label)
        results[ic_col]["discrim"] = analysis_discriminant_validity(df, ic_col, label)
        results[ic_col]["competing"] = analysis_competing_moderators(df, ic_col, label)
        analysis_argument_quality(df, ic_col, label)

    # ── Summary comparison ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY: COSTELLO")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<35s} {'Q400 logit-EV':>15s} {'gpt-4.1-mini':>15s}")
    print(f"  {'-'*66}")
    for key, label in [
        ("beta_ic2", "β_IC² (quadratic)"),
        ("p_ic2", "p(β_IC²)"),
        ("dr2", "ΔR² (quad vs linear)"),
        ("p", "p(ΔR²)"),
        ("r2_quad", "Full quadratic R²"),
    ]:
        v1 = results["IC_q400_logit"]["quad"][key]
        v2 = results["IC_openai_pre"]["quad"][key]
        if key.startswith("p"):
            print(f"  {label:<35s} {fmt_p(v1):>15s} {fmt_p(v2):>15s}")
        else:
            print(f"  {label:<35s} {v1:>15.4f} {v2:>15.4f}")

    for key, label in [
        ("dr2", "Discriminant ΔR²"),
        ("p", "Discriminant p"),
    ]:
        v1 = results["IC_q400_logit"]["discrim"][key]
        v2 = results["IC_openai_pre"]["discrim"][key]
        if key == "p":
            print(f"  {label:<35s} {fmt_p(v1):>15s} {fmt_p(v2):>15s}")
        else:
            print(f"  {label:<35s} {v1:>15.4f} {v2:>15.4f}")


if __name__ == "__main__":
    main()
