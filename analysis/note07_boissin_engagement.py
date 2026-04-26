#!/usr/bin/env python3
"""note07_boissin_engagement.py — SI Note 7.

Within-Boissin engagement-independence test: does the linear-resistance
signal moderate with dialogue word count? Reproduces SI Table tab:boissin_wc.

Result claim (SI Note 7): all four BFs ≤ 0.40; β(IC × wc) is directionally
negative (longer conversations attenuate resistance, the opposite of what
an engagement-driven account would predict).
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "boissin2025"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def fit_separate(sub):
    """Two separate moderation tests on Boissin (raw IC scale, paper convention):
      Test 1 (linear-IC × wc):  y ~ IC + wc + Pre + IC×wc  vs  y ~ IC + wc + Pre
      Test 2 (quadratic-IC × wc): y ~ IC + IC² + wc + Pre + IC²×wc  vs  y ~ IC + IC² + wc + Pre
    """
    y = sub["belief_change"].values.astype(float)
    ic_raw = sub["IC"].values.astype(float)
    ic = zs(ic_raw)
    ic2 = zs(ic_raw ** 2)
    wc = zs(sub["wc_all"].values)
    pre = zs(sub["PreBelief"].values)

    # Test 1: linear IC × wc
    X1_red = sm.add_constant(np.column_stack([ic, wc, pre]))
    X1_full = sm.add_constant(np.column_stack([ic, wc, pre, ic * wc]))
    m1_red = sm.OLS(y, X1_red).fit()
    m1_full = sm.OLS(y, X1_full).fit()
    bf_icxwc = float(np.exp((m1_red.bic - m1_full.bic) / 2))

    # Test 2: quadratic IC × wc
    X2_red = sm.add_constant(np.column_stack([ic, ic2, wc, pre]))
    X2_full = sm.add_constant(np.column_stack([ic, ic2, wc, pre, ic2 * wc]))
    m2_red = sm.OLS(y, X2_red).fit()
    m2_full = sm.OLS(y, X2_full).fit()
    bf_ic2xwc = float(np.exp((m2_red.bic - m2_full.bic) / 2))

    return dict(
        n=len(sub),
        b_icxwc=m1_full.params[4], p_icxwc=m1_full.pvalues[4], bf_icxwc=bf_icxwc,
        b_ic2xwc=m2_full.params[5], p_ic2xwc=m2_full.pvalues[5], bf_ic2xwc=bf_ic2xwc,
    )


def main():
    an = pd.read_csv(DATA / "analysis_data.csv")
    rows = [json.loads(l) for l in open(DATA_Q / "boissin_texts_for_scoring_qwenorpo400.jsonl")]
    # Word count of full-dialogue text_all
    ic = pd.DataFrame({
        "participantId": [r["participantId"] for r in rows],
        "IC":     [r["ic_qwenorpo400_all_logit"] for r in rows],
        "wc_all": [len(str(r["text_all"]).split()) for r in rows],
    })
    df = an.merge(ic, on="participantId", how="left")
    for c in ["belief_change", "PreBelief", "IC", "wc_all"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["belief_change", "IC", "wc_all", "PreBelief"]).copy()

    print(f"SI Note 7 — Within-Boissin engagement-independence (n = {len(df)})\n")
    print(f"{'Cell':<35s} {'Effect':<18s} {'β':>8s}  {'p':>7s}  {'BF₁₀':>6s}")
    print("-" * 90)

    # Pooled
    r = fit_separate(df)
    print(f"{'Pooled':<35s} {'IC × wc':<18s} {r['b_icxwc']:>+8.2f}  {fmt_p(r['p_icxwc']):>7s}  {r['bf_icxwc']:>6.2f}")
    print(f"{'Pooled':<35s} {'IC² × wc':<18s} {r['b_ic2xwc']:>+8.2f}  {fmt_p(r['p_ic2xwc']):>7s}  {r['bf_ic2xwc']:>6.2f}")

    # AI × Human-like cell
    aihl = df[(df["Speaker"] == "AI") & (df["PromptType"] == "Human-like")].copy()
    r = fit_separate(aihl)
    print(f"{'AI × Human-like':<35s} {'IC × wc':<18s} {r['b_icxwc']:>+8.2f}  {fmt_p(r['p_icxwc']):>7s}  {r['bf_icxwc']:>6.2f}")
    print(f"{'AI × Human-like':<35s} {'IC² × wc':<18s} {r['b_ic2xwc']:>+8.2f}  {fmt_p(r['p_ic2xwc']):>7s}  {r['bf_ic2xwc']:>6.2f}")


if __name__ == "__main__":
    main()
