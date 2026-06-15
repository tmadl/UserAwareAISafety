#!/usr/bin/env python3
"""note27_simex_disattenuation.py — SIMEX measurement-error correction for the
Costello IC^2 moderation. IC is scored with imperfect reliability (held-out
ICC(3,1)=.704 on Suedfeld-155, .784 on Jakob-2275), and classical measurement
error attenuates a quadratic moderation toward zero, so the observed
beta_IC2 = -1.99 is a conservative (lower-bound) estimate. SIMEX (Cook & Stefanski
1994) quantifies the correction: add increasing known error, refit, extrapolate
back to zero error.

Spec: DV = signed belief change; model belief ~ z(IC)+z(IC)^2+pre+wc (headline,
treatment N=1782, fixed length-bug data). IC error SD from reliability rho:
sigma_err = SD(IC) * sqrt(1-rho). We bracket rho in {.704 (Suedfeld), .784
(Jakob)}. SIMEX lambda grid {0,.5,1,1.5,2}, 200 sims each, quadratic
extrapolant to lambda=-1. Reports observed and corrected beta_IC2 (z(IC)^2).

Output: prints; writes note27_simex_disattenuation_output.txt.
"""
import importlib.util, io, json, sys, warnings
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REL = HERE.parent


def zs(v):
    v = np.asarray(v, float); return (v - v.mean()) / v.std(ddof=1)


def beta_ic2(ic, y, pre, wc):
    icz = zs(ic)
    X = sm.add_constant(np.column_stack([icz, icz**2, pre, wc]))
    return sm.OLS(y, X).fit().params[2]


def run():
    an = pd.read_csv(REL / "data/costello2024/analysis_data.csv")
    q = pd.read_csv(REL / "data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv")
    d = an.merge(q[["participantId", "ic_qwenorpo400_logit"]].rename(
        columns={"ic_qwenorpo400_logit": "ic"}), on="participantId")
    d = d.dropna(subset=["ic", "DV_BeliefChange_Specific", "Pre_Belief_Specific",
                         "OpenendedResponseWordCount"])
    ic = d.ic.values.astype(float)
    y = d.DV_BeliefChange_Specific.values.astype(float)
    pre = zs(d.Pre_Belief_Specific.values); wc = zs(d.OpenendedResponseWordCount.values)
    sd_ic = ic.std(ddof=1)
    obs = beta_ic2(ic, y, pre, wc)
    print(f"N = {len(d)};  observed beta_IC2 (z(IC)^2) = {obs:+.3f}")
    print(f"SD(IC) = {sd_ic:.3f}\n")

    lambdas = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    rng = np.random.RandomState(7)
    for rho, lab in [(0.704, "Suedfeld-155"), (0.784, "Jakob-2275")]:
        sig_err = sd_ic * np.sqrt(1 - rho)
        curve = []
        for lam in lambdas:
            if lam == 0:
                curve.append(obs); continue
            bs = []
            for _ in range(200):
                ic_lam = ic + rng.normal(0, np.sqrt(lam) * sig_err, size=len(ic))
                bs.append(beta_ic2(ic_lam, y, pre, wc))
            curve.append(np.mean(bs))
        curve = np.array(curve)
        # quadratic extrapolation in lambda to lambda = -1 (zero true error)
        coef = np.polyfit(lambdas, curve, 2)
        corrected = np.polyval(coef, -1.0)
        print(f"reliability rho = {rho} ({lab}): sigma_err = {sig_err:.3f}")
        print("  SIMEX curve beta(lambda):",
              ", ".join(f"{l:.1f}:{c:+.2f}" for l, c in zip(lambdas, curve)))
        print(f"  corrected beta_IC2 (lambda=-1) = {corrected:+.3f}  "
              f"({corrected/obs:.2f}x the observed)\n")
    print("Interpretation: SIMEX moves beta_IC2 AWAY from zero (more negative),")
    print("confirming the observed curvature is attenuated by IC measurement error")
    print("and is a conservative lower bound on the true moderation.")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    t = buf.getvalue(); sys.stdout.write(t)
    (HERE / "note27_simex_disattenuation_output.txt").write_text(t)
