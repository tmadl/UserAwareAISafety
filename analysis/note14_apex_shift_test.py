#!/usr/bin/env python3
"""note14_apex_shift_test.py — SI Note 14.

Within-Costello apex-shift test: does the inverted-U apex shift rightward
(toward higher IC) when the AI dialogue is longer? Tests two ways:
  (1) tertile splits on full-dialogue word count, each tertile fit independently
  (2) omnibus interaction: IC + IC² + log(wc) + IC×log(wc) [+ IC²×log(wc)]

Result claim (SI Note 14): apex shift NOT supported. Apex locations across
tertiles are 3.09 / 2.73 / 3.40 — not monotonic. Adding IC × log(wc) gives
nominally-significant β = -1.78 (p = .010) but BF₁₀ = 0.67 (BIC penalises);
adding IC²×log(wc) further worsens fit (BF₁₀ = 0.047).
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


def fit_quad_apex_ci(sub, n_boot=2000):
    """Fit y ~ IC + IC² + log(wc) + Pre. Note 14 reports β on the RAW IC²
    scale (matches the paper's raw-IC² convention for tertile / placebo
    tables). Bootstrap apex CI on raw IC."""
    y = sub["DV_BeliefChange_Specific"].values.astype(float)
    raw = sub["IC"].values.astype(float)
    log_wc = zs(np.log(sub["wc_all"].values.astype(float)))
    pre = zs(sub["Pre_Belief_Specific"].values)
    X = sm.add_constant(np.column_stack([raw, raw ** 2, log_wc, pre]))
    m = sm.OLS(y, X).fit()
    b1, b2 = m.params[1], m.params[2]
    apex = -b1 / (2 * b2) if abs(b2) > 1e-12 else np.nan
    rng = np.random.default_rng(42)
    peaks = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rb = raw[idx]
        try:
            Xb = sm.add_constant(np.column_stack([
                rb, rb ** 2,
                zs(np.log(sub["wc_all"].values.astype(float)[idx])),
                zs(sub["Pre_Belief_Specific"].values[idx]),
            ]))
            mb = sm.OLS(y[idx], Xb).fit()
            peaks.append(-mb.params[1] / (2 * mb.params[2]))
        except Exception:
            pass
    lo, hi = np.percentile(peaks, [2.5, 97.5])
    return dict(n=len(sub), beta=m.params[2], p=m.pvalues[2],
                apex=apex, ci_lo=lo, ci_hi=hi)


def main():
    an = pd.read_csv(DATA / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC":     q["ic_qwenorpo400_logit"].astype(float).values,
        "wc_all": [len(str(m["text_all"]).split()) for m in meta],
    })
    df = an.merge(ic, on="participantId", how="left")
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific", "IC", "wc_all"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DV_BeliefChange_Specific", "IC", "wc_all", "Pre_Belief_Specific"]).copy()
    df = df[df["wc_all"] > 0].copy()  # for log

    # === Tertile splits on wc_all ===
    print("SI Note 14 — Within-Costello apex-shift test\n")
    print("=== Tertile splits on full-dialogue word count (wc_all) ===")
    qs = df["wc_all"].quantile([1/3, 2/3]).values
    df["tertile"] = pd.cut(df["wc_all"], bins=[-1, qs[0], qs[1], df["wc_all"].max() + 1],
                           labels=["low", "mid", "high"])
    print(f"{'Tertile':<8s} {'wc range':<18s} {'n':>5s}  {'β_IC²':>8s}  {'p':>8s}  {'Peak':>5s}  {'95% CI':>16s}")
    print("-" * 80)
    for label, group in df.groupby("tertile", observed=True):
        wc_min, wc_max = int(group["wc_all"].min()), int(group["wc_all"].max())
        r = fit_quad_apex_ci(group)
        print(f"{str(label):<8s} [{wc_min}, {wc_max}]{'':<{16-len(f'[{wc_min}, {wc_max}]')}s}"
              f"{r['n']:>5d}  {r['beta']:>+8.2f}  {fmt_p(r['p']):>8s}  "
              f"{r['apex']:>5.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    # === Omnibus interaction test ===
    print("\n=== Omnibus IC × log(wc) interaction test ===")
    y = df["DV_BeliefChange_Specific"].values.astype(float)
    raw = df["IC"].values.astype(float)
    ic_z, ic2_z = zs(raw), zs(raw ** 2)
    logwc_z = zs(np.log(df["wc_all"].values.astype(float)))
    pre_z = zs(df["Pre_Belief_Specific"].values)
    icxwc = ic_z * logwc_z
    ic2xwc = ic2_z * logwc_z

    X_base = sm.add_constant(np.column_stack([ic_z, ic2_z, logwc_z, pre_z]))
    X_lin = sm.add_constant(np.column_stack([ic_z, ic2_z, logwc_z, pre_z, icxwc]))
    X_full = sm.add_constant(np.column_stack([ic_z, ic2_z, logwc_z, pre_z, icxwc, ic2xwc]))
    m_base = sm.OLS(y, X_base).fit()
    m_lin = sm.OLS(y, X_lin).fit()
    m_full = sm.OLS(y, X_full).fit()
    bf_lin = float(np.exp((m_base.bic - m_lin.bic) / 2))
    bf_full = float(np.exp((m_base.bic - m_full.bic) / 2))

    print(f"{'Model':<55s} {'R²':>6s} {'β(IC×logwc)':>13s} {'p':>7s} {'BF₁₀ vs base':>14s}")
    print(f"{'Base (IC + IC² + log(wc) + PreBelief)':<55s} {m_base.rsquared:>6.4f} "
          f"{'---':>13s} {'---':>7s} {'1':>14s} (reference)")
    print(f"{'+ IC × log(wc)':<55s} {m_lin.rsquared:>6.4f} "
          f"{m_lin.params[5]:>+13.2f} {m_lin.pvalues[5]:>7.3f} {bf_lin:>14.3f}")
    print(f"{'+ IC × log(wc) + IC² × log(wc)':<55s} {m_full.rsquared:>6.4f} "
          f"{m_full.params[5]:>+6.2f} / {m_full.params[6]:>+5.2f} "
          f"{m_full.pvalues[5]:>4.3f}/{m_full.pvalues[6]:.3f} {bf_full:>14.3f}")

    print("\nClaim check (SI Note 14):")
    print("  Apex locations should be ~3.09 / 2.73 / 3.40 (not monotonic) — see tertile table above.")
    print("  IC × log(wc) BF₁₀ ≈ 0.67 (BIC penalises the interaction).")
    print("  Adding IC² × log(wc) further attenuates: BF₁₀ ≈ 0.047 (worsens fit).")


if __name__ == "__main__":
    main()
