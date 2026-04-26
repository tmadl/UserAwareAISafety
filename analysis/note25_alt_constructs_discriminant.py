#!/usr/bin/env python3
"""note25_alt_constructs_discriminant.py — SI Note 25.

Discriminant validity against alternative text-derived construct moderators
(AOT, IH, NFC, OMI). Reproduces:
  - tab:alt_constructs (standalone quadratic for each construct + IC reference)
  - tab:alt_constructs_partialled (residual quadratic after partialling on IC)
  - the validated text-IH rows (Guo-EMNLP-anchored scorer)
  - the rubric-IH-vs-validated-IH comparison: rubric IH residual BF=285,
    validated IH residual BF=0.58 (491× attenuation)
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


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def fit_standalone(df, x_col):
    """Standalone quadratic moderation: DV ~ z(X) + z(X²) + z(pre) + z(wc).
    Returns β_lin, β_quad, p, BF, apex(raw)."""
    sub = df.dropna(subset=["DV_BeliefChange_Specific", x_col,
                            "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()
    y = sub["DV_BeliefChange_Specific"].values.astype(float)
    raw = sub[x_col].values.astype(float)
    x_z, x2_z = zs(raw), zs(raw ** 2)
    pre = zs(sub["Pre_Belief_Specific"].values)
    wc = zs(sub["OpenendedResponseWordCount"].values)
    X_lin = sm.add_constant(np.column_stack([x_z, pre, wc]))
    X_quad = sm.add_constant(np.column_stack([x_z, x2_z, pre, wc]))
    m_lin = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()
    bf = float(np.exp((m_lin.bic - m_quad.bic) / 2))
    sd_x, sd_sq = np.std(raw), np.std(raw ** 2)
    b1, b2 = m_quad.params[1], m_quad.params[2]
    apex = -b1 * sd_sq / (2 * b2 * sd_x) if abs(b2) > 1e-12 else np.nan
    return dict(n=len(sub), b_lin=m_quad.params[1], p_lin=m_quad.pvalues[1],
                b_quad=m_quad.params[2], p_quad=m_quad.pvalues[2],
                bf=bf, apex=apex)


def fit_residual(df, x_col, partial_col):
    """Residual quadratic: residualise X on partial, then fit
    DV ~ z(X|partial) + z((X|partial)²) + z(pre) + z(wc)."""
    sub = df.dropna(subset=["DV_BeliefChange_Specific", x_col, partial_col,
                            "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()
    y = sub["DV_BeliefChange_Specific"].values.astype(float)
    x_z = zs(sub[x_col].values)
    p_z = zs(sub[partial_col].values)
    # Residualise X on partial
    X_partial = sm.add_constant(p_z.reshape(-1, 1))
    resid = x_z - sm.OLS(x_z, X_partial).fit().predict(X_partial)
    res_z, res_sq_z = zs(resid), zs(resid ** 2)
    pre = zs(sub["Pre_Belief_Specific"].values)
    wc = zs(sub["OpenendedResponseWordCount"].values)
    X_lin = sm.add_constant(np.column_stack([res_z, pre, wc]))
    X_quad = sm.add_constant(np.column_stack([res_z, res_sq_z, pre, wc]))
    m_lin = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()
    bf = float(np.exp((m_lin.bic - m_quad.bic) / 2))
    return dict(n=len(sub), b_lin=m_quad.params[1], p_lin=m_quad.pvalues[1],
                b_quad=m_quad.params[2], p_quad=m_quad.pvalues[2], bf=bf)


def main():
    # === Load Costello primary + IC + alt constructs + validated IH ===
    # Note 25 uses two IC reference columns:
    #   - gpt-4.1-mini full-dialogue IC for the rubric-prompted alt constructs
    #     (matches their scoring window and channel: same gpt-4.1-mini logit-EV pipeline)
    #   - Q400 logit-EV (paper primary) for the validated IH residualisation
    #     (matches the main-text headline framing)
    an = pd.read_csv(DATA / "analysis_data.csv")
    ic_old = pd.read_csv(DATA / "all_complexity_scores.csv").rename(
        columns={"IC_openai": "IC"})  # full-dialogue gpt-4.1-mini
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic_q400 = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC_q400": q["ic_qwenorpo400_logit"].astype(float).values,
    })
    alt = pd.read_csv(DATA / "alt_constructs_logitev.csv")[
        ["participantId", "AOT_ev", "IH_ev", "NFC_ev", "OMI_ev"]
    ].rename(columns={"AOT_ev": "AOT_text", "IH_ev": "IH_rubric",
                      "NFC_ev": "NFC_text", "OMI_ev": "OMI_text"})
    ih_val = pd.read_csv(DATA / "ih_aot_prototypes/preds_ih_guo_only_decomp_ckpt100.csv")
    ih_val = ih_val[ih_val["has_text"] == True].groupby(
        "participantId", as_index=False)["pred_ih_ev"].mean().rename(
        columns={"pred_ih_ev": "IH_validated"})

    df = (an.merge(ic_old[["participantId", "IC"]], on="participantId", how="inner")
            .merge(ic_q400, on="participantId", how="left")
            .merge(alt, on="participantId", how="left")
            .merge(ih_val, on="participantId", how="left"))
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "IC", "IC_q400",
              "AOT_text", "IH_rubric", "NFC_text", "OMI_text", "IH_validated"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # === Table tab:alt_constructs — standalone quadratic per construct ===
    print("SI Note 25 — Discriminant validity against alternative constructs\n")
    print("=== Table tab:alt_constructs (standalone quadratic) ===")
    print(f"{'Construct':<40s} {'n':>5s}  {'β_lin':>8s}  {'p':>8s}  {'β_quad':>8s}  {'p':>8s}  {'apex':>5s}  {'BF₁₀':>10s}")
    print("-" * 100)
    for label, col in [("IC (gpt-4.1-mini) [reference]",        "IC"),
                       ("AOT (gpt-4.1-mini rubric)",            "AOT_text"),
                       ("IH  (gpt-4.1-mini rubric)",            "IH_rubric"),
                       ("IH  (Guo-validated, r_Guo = .71)",     "IH_validated"),
                       ("NFC (gpt-4.1-mini rubric)",            "NFC_text"),
                       ("OMI (gpt-4.1-mini rubric)",            "OMI_text")]:
        r = fit_standalone(df, col)
        # Apex meaningful only when curvature significant
        apex_str = f"{r['apex']:.2f}" if r['p_quad'] < .05 else "n.s."
        print(f"{label:<40s} {r['n']:>5d}  {r['b_lin']:>+8.2f}  {fmt_p(r['p_lin']):>8s}  "
              f"{r['b_quad']:>+8.2f}  {fmt_p(r['p_quad']):>8s}  {apex_str:>5s}  {r['bf']:>10.2g}")

    # === Table tab:alt_constructs_partialled — residual quadratic ===
    print("\n=== Table tab:alt_constructs_partialled (residual after partialling on IC) ===")
    print(f"{'Residual construct':<40s} {'n':>5s}  {'β_lin':>8s}  {'p':>8s}  {'β_quad':>8s}  {'p':>8s}  {'BF₁₀':>10s}")
    print("-" * 95)
    for label, col, partial_col in [
        ("AOT | IC (gpt-4.1-mini)",                 "AOT_text",     "IC"),
        ("IH (gpt-4.1-mini rubric) | IC (gpt-mini)", "IH_rubric",   "IC"),
        ("IH (Guo-validated)       | IC (Q400)",     "IH_validated", "IC_q400"),
        ("NFC | IC (gpt-4.1-mini)",                 "NFC_text",     "IC"),
        ("OMI | IC (gpt-4.1-mini)",                 "OMI_text",     "IC"),
    ]:
        r = fit_residual(df, col, partial_col)
        print(f"{label:<43s} {r['n']:>5d}  {r['b_lin']:>+8.2f}  {fmt_p(r['p_lin']):>8s}  "
              f"{r['b_quad']:>+8.2f}  {fmt_p(r['p_quad']):>8s}  {r['bf']:>10.2g}")

    # Convergence of validated IH with self-report IH (Study 1 subset)
    s1 = df[(df["StudyNumber"] == 1)].dropna(subset=["IH_validated", "IH"]).copy()
    if len(s1) > 0:
        s1["IH"] = pd.to_numeric(s1["IH"], errors="coerce")
        s1 = s1.dropna(subset=["IH"])
        from scipy.stats import pearsonr
        r_pearson, p_pearson = pearsonr(s1["IH_validated"], s1["IH"])
        print(f"\nConvergence (Study 1, n = {len(s1)}):")
        print(f"  validated text-IH ~ Costello self-report IH: Pearson r = {r_pearson:.3f}, "
              f"p = {fmt_p(p_pearson)}")


if __name__ == "__main__":
    main()
