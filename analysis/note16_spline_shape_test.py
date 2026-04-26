#!/usr/bin/env python3
"""note16_spline_shape_test.py — SI Note 16.

Penalised-spline / cubic-OLS shape test for the Costello inverted-U.
Reproduces SI Table tab:shape_test (linear / quadratic / cubic AIC/BIC/R²
plus BF₁₀ vs simpler) and the asymmetry diagnostic on the GAM smooth.

Note: GAM portion requires `pygam`. If pygam isn't installed, the cubic
OLS comparison still runs and the GAM section is skipped.
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


def main():
    an = pd.read_csv(DATA / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC": q["ic_qwenorpo400_logit"].astype(float).values,
    })
    df = an.merge(ic, on="participantId", how="left")
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific", "OpenendedResponseWordCount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    sub = df.dropna(subset=["DV_BeliefChange_Specific", "IC",
                            "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()
    sub = sub[sub["OpenendedResponseWordCount"] > 0]   # for log
    y = sub["DV_BeliefChange_Specific"].values.astype(float)

    # Note 16 uses LOG word count (z-scored) + pre-belief (z-scored)
    ic_z = zs(sub["IC"].values)
    log_wc = zs(np.log(sub["OpenendedResponseWordCount"].values.astype(float)))
    pre_z = zs(sub["Pre_Belief_Specific"].values)

    # === Linear / Quadratic / Cubic OLS ===
    X_lin = sm.add_constant(np.column_stack([ic_z, log_wc, pre_z]))
    X_quad = sm.add_constant(np.column_stack([ic_z, ic_z ** 2, log_wc, pre_z]))
    X_cubic = sm.add_constant(np.column_stack([ic_z, ic_z ** 2, ic_z ** 3, log_wc, pre_z]))

    m_lin = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()
    m_cubic = sm.OLS(y, X_cubic).fit()

    bf_q_vs_l = float(np.exp((m_lin.bic - m_quad.bic) / 2))
    bf_c_vs_q = float(np.exp((m_quad.bic - m_cubic.bic) / 2))

    print("SI Note 16 — Shape test: linear / quadratic / cubic / GAM (Costello)")
    print(f"Sample: n = {len(sub)} (log word count covariate; z-scored)\n")
    print(f"{'Model':<35s} {'AIC':>10s} {'BIC':>10s} {'R²':>6s} {'BF₁₀ vs simpler':>18s}")
    print("-" * 85)
    print(f"{'Linear (IC + covars)':<35s} {m_lin.aic:>10.1f} {m_lin.bic:>10.1f} "
          f"{m_lin.rsquared:>6.3f} {'---':>18s}")
    print(f"{'Quadratic (IC, IC² + covars)':<35s} {m_quad.aic:>10.1f} {m_quad.bic:>10.1f} "
          f"{m_quad.rsquared:>6.3f} {bf_q_vs_l:>18.1f}")
    print(f"{'Cubic (IC, IC², IC³ + covars)':<35s} {m_cubic.aic:>10.1f} {m_cubic.bic:>10.1f} "
          f"{m_cubic.rsquared:>6.3f} {bf_c_vs_q:>18.3f}")

    # Cubic IC³ p-value (SI quotes p=.063)
    p_ic3 = m_cubic.pvalues[3]
    dr2_cubic = m_cubic.rsquared - m_quad.rsquared
    print(f"\nCubic IC³ term: p = {p_ic3:.3f}; ΔR² (cubic over quadratic) = {dr2_cubic:.4f}")

    # Quadratic apex on raw IC (this Note's spec — log_wc covariate)
    raw = sub["IC"].values.astype(float)
    sd_ic = np.std(raw); sd_sq = np.std(raw ** 2)
    # In log-wc spec: convert z(IC²) coefficient back to apex on raw IC
    # m_quad uses ic_z and ic_z**2 (NOT zs(ic²)) — different from canonical paper spec
    # apex on raw IC: dy/dIC = b1/sd + 2*b2*(IC-mean)/sd²
    b1, b2 = m_quad.params[1], m_quad.params[2]
    apex_raw = np.mean(raw) - b1 * sd_ic / (2 * b2)
    print(f"\nQuadratic apex (this Note's log-wc spec): IC = {apex_raw:.2f}")

    # === GAM (optional) ===
    try:
        from pygam import LinearGAM, s, l
        gam = LinearGAM(s(0, n_splines=15) + l(1) + l(2)).fit(
            np.column_stack([sub["IC"].values, log_wc, pre_z]), y
        )
        r2_gam = 1 - np.sum((y - gam.predict(np.column_stack(
            [sub["IC"].values, log_wc, pre_z]))) ** 2) / np.sum((y - y.mean()) ** 2)
        # Apex on smooth: predict on grid with log_wc, pre_z held at sample means
        ic_grid = np.linspace(raw.min(), raw.max(), 200)
        Xg = np.column_stack([ic_grid, np.zeros_like(ic_grid), np.zeros_like(ic_grid)])
        y_smooth = gam.predict(Xg)
        apex_gam = ic_grid[np.argmax(y_smooth)]
        # Asymmetry diagnostic
        ymax = np.max(y_smooth)
        # Lowest 10% on left/right of apex
        left_mask = ic_grid <= np.percentile(raw, 10)
        right_mask = ic_grid >= np.percentile(raw, 90)
        left_drop = ymax - np.mean(y_smooth[left_mask])
        right_drop = ymax - np.mean(y_smooth[right_mask])
        ratio = right_drop / left_drop if left_drop > 0 else float('nan')

        print(f"\n{'GAM (spline on IC + linear covars)':<35s} {'---':>10s} {'---':>10s} "
              f"{r2_gam:>6.3f} {'(EDoF-based)':>18s}")
        print(f"\nAsymmetry diagnostic:")
        print(f"  GAM apex (raw IC):                   {apex_gam:.2f}")
        print(f"  Apex-to-left-10%-tail descent:       {left_drop:.1f} DV-points")
        print(f"  Apex-to-right-10%-tail descent:      {right_drop:.1f} DV-points")
        print(f"  Right/left ratio (1.0 = symmetric):  {ratio:.2f}")
    except ImportError:
        print("\nGAM section skipped — install pygam to reproduce: pip install pygam")


if __name__ == "__main__":
    main()
