#!/usr/bin/env python3
"""note16b_right_arm.py — formal defense (or honest demotion) of the
high-IC arm of the inverted-U (reviewer M3: "the right arm rests on sparse,
leverage-prone data; a monotone-plateau account may suffice").

Spec declaration: DV = DV_BeliefChange_Specific (signed, pre minus post),
treatment arm only (N=1,782), IC = IC_q400_logit (pre-treatment, primary),
covariates z(pre-belief) + z(word count) throughout (canonical spec);
OLS; z(IC)^2 parameterisation; BFs from BIC differences.

Tests:
  (0) positive control: reproduce headline quadratic beta=-1.97, p<.001.
  (1) Q5 vs Q2-Q4 contrast (rank quintiles): does the top quintile revise
      less than the mid quintiles, covariate-adjusted? One-sided is licensed
      by the directional prediction but two-tailed is reported primary.
  (2) Functional-form race: quadratic vs monotone rise-then-PLATEAU
      (piecewise linear, second slope = 0, knot grid-searched) vs free
      piecewise linear (rise-then-anything) vs linear. BIC comparison.
      If plateau ~ ties quadratic, the yielding-resistance (right-arm
      decline) claim must be softened.
  (3) Influence diagnostics: Cook's distance on the quadratic; refit
      excluding top 1% influence; refits trimming right-tail IC at the
      97.5th percentile and at IC=4.0. Report beta, p, apex stability.

Output: prints; writes note16b_right_arm_output.txt.
"""
import importlib.util
import io
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c01", HERE / "01_costello_analysis.py")
c01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c01)


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


def load():
    df = c01.load_data()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(subset=["IC_q400_logit", "DV_BeliefChange_Specific",
                           "Pre_Belief_Specific",
                           "OpenendedResponseWordCount"]).copy()
    df = df.drop_duplicates("participantId").reset_index(drop=True)
    return df


def fit_ols(y, Xcols):
    X = sm.add_constant(np.column_stack(Xcols))
    return sm.OLS(y, X).fit()


def run():
    df = load()
    y = df["DV_BeliefChange_Specific"].values
    ic_raw = df["IC_q400_logit"].values
    icz = zs(ic_raw)
    pre = zs(df["Pre_Belief_Specific"].values)
    wc = zs(df["OpenendedResponseWordCount"].values)
    n = len(df)
    print(f"N = {n} (treatment arm, complete case)")

    # (0) positive control
    m_quad = fit_ols(y, [icz, icz**2, pre, wc])
    m_lin = fit_ols(y, [icz, pre, wc])
    bf_quad = float(np.exp((m_lin.bic - m_quad.bic) / 2))
    print(f"(0) positive control: b_quad = {m_quad.params[2]:+.2f} "
          f"(p = {m_quad.pvalues[2]:.2e}), BF_quad-vs-lin = {bf_quad:,.0f} "
          f"[published: -1.97, p<.001, BF 1,086]")
    print()

    # (1) Q5 vs Q2-Q4 contrast
    q = pd.qcut(pd.Series(ic_raw).rank(method="first"), 5, labels=False).values
    sub = q >= 1  # Q2..Q5 only, isolating the right arm from the low tail
    is_q5 = (q[sub] == 4).astype(float)
    mc = fit_ols(y[sub], [is_q5, pre[sub], wc[sub]])
    ci = mc.conf_int()[1]
    print(f"(1) Q5 vs Q2-Q4 (covariate-adjusted, n = {sub.sum()}):")
    print(f"    b(Q5) = {mc.params[1]:+.2f} points, 95% CI "
          f"[{ci[0]:+.2f}, {ci[1]:+.2f}], t = {mc.tvalues[1]:.2f}, "
          f"p = {mc.pvalues[1]:.4f} (two-tailed; one-tailed "
          f"{mc.pvalues[1]/2:.4f} under the directional prediction)")
    # raw (unadjusted) for reference
    mc0 = fit_ols(y[sub], [is_q5])
    print(f"    unadjusted: b(Q5) = {mc0.params[1]:+.2f}, "
          f"p = {mc0.pvalues[1]:.4f}")
    print()

    # (2) functional-form race on raw IC (knots in raw units)
    def bic_of(model):
        return model.bic

    knots = np.quantile(ic_raw, np.linspace(0.15, 0.85, 29))
    best = {}
    for name, free2 in [("plateau (rise-then-flat)", False),
                        ("piecewise free (rise-then-any)", True)]:
        best_bic, best_knot, best_m = np.inf, None, None
        for k in knots:
            x1 = np.minimum(ic_raw, k)        # slope before knot
            x2 = np.maximum(ic_raw - k, 0.0)  # slope after knot
            cols = [zs(x1), zs(x2), pre, wc] if free2 else None
            if free2:
                m = fit_ols(y, [zs(x1), zs(x2), pre, wc])
            else:
                m = fit_ols(y, [zs(x1), pre, wc])
                # plateau: only the pre-knot slope enters; post-knot flat
                # achieved by capping x at the knot
            b = bic_of(m)
            if b < best_bic:
                best_bic, best_knot, best_m = b, k, m
        best[name] = (best_bic, best_knot, best_m)

    m_linear = fit_ols(y, [icz, pre, wc])
    print("(2) functional-form race (BIC; lower = better; knot grid 15th-85th pct):")
    rows = [("linear", m_linear.bic, None),
            ("quadratic (paper)", m_quad.bic, None),
            ("plateau (rise-then-flat)", best["plateau (rise-then-flat)"][0],
             best["plateau (rise-then-flat)"][1]),
            ("piecewise free (rise-then-any)",
             best["piecewise free (rise-then-any)"][0],
             best["piecewise free (rise-then-any)"][1])]
    base = m_quad.bic
    for name, b, k in rows:
        kstr = f" (knot IC = {k:.2f})" if k is not None else ""
        bf_vs_quad = float(np.exp((b - base) / 2))
        print(f"    {name:32s} BIC = {b:9.1f}  "
              f"BF(quad vs this) = {bf_vs_quad:8.2f}{kstr}")
    mf = best["piecewise free (rise-then-any)"][2]
    print(f"    piecewise-free post-knot slope: b = {mf.params[2]:+.2f} "
          f"(p = {mf.pvalues[2]:.4f}; negative = decline after knot)")
    print()

    # (3) influence + right-tail trimming
    infl = m_quad.get_influence()
    cooks = infl.cooks_distance[0]
    thr = np.quantile(cooks, 0.99)
    keep = cooks < thr
    icz_k = zs(ic_raw[keep])
    mk = fit_ols(y[keep], [icz_k, icz_k**2, zs(df["Pre_Belief_Specific"].values[keep]),
                           zs(df["OpenendedResponseWordCount"].values[keep])])
    print(f"(3) influence/trimming:")
    print(f"    drop top 1% Cook's D (n = {keep.sum()}): "
          f"b_quad = {mk.params[2]:+.2f} (p = {mk.pvalues[2]:.4f})")
    for label, mask in [("IC < 97.5th pct", ic_raw < np.quantile(ic_raw, 0.975)),
                        ("IC < 4.0", ic_raw < 4.0)]:
        iczt = zs(ic_raw[mask])
        mt = fit_ols(y[mask], [iczt, iczt**2,
                               zs(df["Pre_Belief_Specific"].values[mask]),
                               zs(df["OpenendedResponseWordCount"].values[mask])])
        m_lin_t = fit_ols(y[mask], [iczt,
                                    zs(df["Pre_Belief_Specific"].values[mask]),
                                    zs(df["OpenendedResponseWordCount"].values[mask])])
        bf_t = float(np.exp((m_lin_t.bic - mt.bic) / 2))
        # apex on raw scale
        b1, b2 = mt.params[1], mt.params[2]
        mu, sd = ic_raw[mask].mean(), ic_raw[mask].std(ddof=1)
        apex = mu + sd * (-b1 / (2 * b2)) if b2 != 0 else np.nan
        print(f"    trim {label} (n = {mask.sum()}): b_quad = {mt.params[2]:+.2f} "
              f"(p = {mt.pvalues[2]:.4f}), BF = {bf_t:,.1f}, apex = {apex:.2f}")
    # data support: IC percentiles for figure truncation decision
    pcts = np.percentile(ic_raw, [1, 2.5, 97.5, 99])
    print(f"    IC percentiles: 1st = {pcts[0]:.2f}, 2.5th = {pcts[1]:.2f}, "
          f"97.5th = {pcts[2]:.2f}, 99th = {pcts[3]:.2f}, "
          f"max = {ic_raw.max():.2f}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    (HERE / "note16b_right_arm_output.txt").write_text(text)
    sys.stdout.write(f"\nWrote {HERE / 'note16b_right_arm_output.txt'}\n")
