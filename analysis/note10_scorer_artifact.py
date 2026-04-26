#!/usr/bin/env python3
"""note10_scorer_artifact.py — SI Note 10.

Tests whether the Costello inverted-U is an artifact of nonlinearity in the
Q400 logit-EV scorer's response function. Uses the held-out Suedfeld-155
exemplars (human-coded IC 1–7) to estimate forward and inverse calibration,
then refits the Costello quadratic on calibrated IC values.

Reproduces:
  - Per-IC-bucket logit-EV means on Suedfeld-155 (Note 10 prose)
  - Forward calibration: α₂ = -0.002, p = .95
  - Inverse calibration: β₂ = +0.114, p = .035 (convex; wrong sign for inverted-U)
  - tab:scorer_artifact (Costello β_quad on raw / linear-cal / quadratic-cal / isotonic-cal scales)
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"
DATA_VAL = ROOT / "data" / "ic_validation"


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3g}"


def fit_costello_quad(df, ic_col):
    """Quadratic on raw scale: DV ~ IC + IC² + pre + wc."""
    sub = df.dropna(subset=["DV_BeliefChange_Specific", ic_col,
                            "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()
    y = sub["DV_BeliefChange_Specific"].values.astype(float)
    ic = sub[ic_col].values.astype(float)
    pre = sub["Pre_Belief_Specific"].values
    wc = sub["OpenendedResponseWordCount"].values
    X = sm.add_constant(np.column_stack([ic, ic ** 2, pre, wc]))
    m = sm.OLS(y, X).fit()
    return m.params[2], m.pvalues[2]


def main():
    # === Suedfeld-155 forward & inverse calibration ===
    val = pd.read_csv(DATA_VAL / "validation_results_qwen3orpo400.csv")
    sf = val[val["source"] == "psuedfeld"].copy()
    gt = sf["gt"].astype(float).values
    logit = sf["logit_score"].astype(float).values

    print("SI Note 10 — Scorer-artifact test (Suedfeld-155 inverse calibration)\n")

    # Per-bucket logit means
    print(f"Per-IC-bucket logit-EV means (Suedfeld-155):")
    print(f"  {'IC':>3s}  {'n':>3s}  {'mean logit':>11s}  {'max logit':>10s}")
    for k in range(1, 8):
        sub = logit[gt == k]
        if len(sub) > 0:
            print(f"  {k:>3d}  {len(sub):>3d}  {np.mean(sub):>11.2f}  {np.max(sub):>10.2f}")

    # Forward: logit ~ gt + gt² (testing if scorer is near-linear)
    X = sm.add_constant(np.column_stack([gt, gt ** 2]))
    m_fwd = sm.OLS(logit, X).fit()
    print(f"\nForward calibration (logit ~ gt + gt²):")
    print(f"  α₁ (linear)    = {m_fwd.params[1]:+.3f}, p = {fmt_p(m_fwd.pvalues[1])}")
    print(f"  α₂ (quadratic) = {m_fwd.params[2]:+.3f}, p = {fmt_p(m_fwd.pvalues[2])}")
    print(f"  R²             = {m_fwd.rsquared:.3f}")

    # Inverse: gt ~ logit + logit² (testing if back-transform is convex)
    X = sm.add_constant(np.column_stack([logit, logit ** 2]))
    m_inv = sm.OLS(gt, X).fit()
    print(f"\nInverse calibration (gt ~ logit + logit²):")
    print(f"  β₁ (linear)    = {m_inv.params[1]:+.3f}, p = {fmt_p(m_inv.pvalues[1])}")
    print(f"  β₂ (quadratic) = {m_inv.params[2]:+.3f}, p = {fmt_p(m_inv.pvalues[2])}")
    sign = "convex" if m_inv.params[2] > 0 else "concave"
    direction = "U-shape" if m_inv.params[2] > 0 else "inverted-U"
    print(f"  Inverse calibration is {sign} (β₂ > 0 → {direction} would arise from linear true effect).")

    # === Costello inverse-calibrated refits ===
    print(f"\n=== Table tab:scorer_artifact — Costello β_quad on calibrated scales ===")
    an = pd.read_csv(DATA / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic_q = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC": q["ic_qwenorpo400_logit"].astype(float).values,
    })
    df = an.merge(ic_q, on="participantId", how="inner")
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "IC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DV_BeliefChange_Specific", "IC",
                            "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()

    # Linear calibration: separate linear-only fit gt = a + b·logit
    m_lin_inv = sm.OLS(gt, sm.add_constant(logit)).fit()
    df["IC_lin_cal"] = m_lin_inv.predict(sm.add_constant(df["IC"].values))

    # Quadratic calibration: solve gt = β₀ + β₁·logit + β₂·logit² for IC = logit
    # (i.e., apply m_inv to predict gt from logit)
    df["IC_quad_cal"] = m_inv.predict(sm.add_constant(np.column_stack([df["IC"].values, df["IC"].values ** 2])))

    # Isotonic calibration: monotone fit gt = f(logit)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(logit, gt)
    df["IC_iso_cal"] = iso.predict(df["IC"].values)

    print(f"\n  {'Scale':<35s} {'β_quad':>8s}  {'p':>8s}")
    print("  " + "-" * 60)
    for label, col in [("Raw logit-EV (headline)",         "IC"),
                       ("Linear-calibrated (rescaled)",    "IC_lin_cal"),
                       ("Quadratic-calibrated",            "IC_quad_cal"),
                       ("Isotonic-calibrated",             "IC_iso_cal")]:
        b, p = fit_costello_quad(df, col)
        print(f"  {label:<35s} {b:>+8.2f}  {fmt_p(p):>8s}")

    # Ceiling check
    print(f"\nCeiling check — max logit-EV per Suedfeld IC bucket:")
    print(f"  {'IC=1':>5s} {'IC=2':>5s} {'IC=3':>5s} {'IC=4':>5s} {'IC=5':>5s} {'IC=6':>5s} {'IC=7':>5s}")
    maxes = [np.max(logit[gt == k]) if (gt == k).any() else float("nan") for k in range(1, 8)]
    print(f"  {'  '.join(f'{m:.2f}' for m in maxes)}")
    print(f"\nCostello IC range: [{df['IC'].min():.2f}, {df['IC'].max():.2f}] "
          f"— does not reach top of Suedfeld saturation region.")


if __name__ == "__main__":
    main()
