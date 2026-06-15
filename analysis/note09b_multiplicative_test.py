#!/usr/bin/env python3
"""note09b_multiplicative_test.py — does the treatment-arm |Delta| curvature
exceed what pure multiplicative scaling of the control-arm curvature predicts?
(Reviewer M2: control |Delta| mean is ~3.3x smaller and its raw quadratic
~3.5x smaller -- proportionally identical, so a multiplicative account
"curvature scales with movement magnitude" may fit without any differential
moderation. The gamma-log GLM that would test this was reported for the
treatment arm only.)

Logic: if the IC^2 curvature were purely multiplicative with the arm mean,
a log link would render the two arms' curvature EQUAL (proportional on the
raw scale = additive/equal on the log scale). So the discriminating test is
the arm x z(IC)^2 interaction in a Gamma(log) GLM on |Delta|:
  - interaction ~ 0  => consistent with pure multiplicative scaling (M2 stands)
  - treatment curvature more negative on log scale => differential moderation
    beyond magnitude scaling (M2 answered).

Spec declaration:
  DV   = |DV_BeliefChange_Specific| (pre-post abs belief change).
         Gamma needs strictly positive support; we add +1 (belief scale is
         0-100, so +1 is negligible) and also report a |Delta|>0 subset.
  IC   = ic_q400 (primary; treatment scored on pre-treatment text,
         control on conRestatement -- a scoring-window asymmetry that
         confounds any arm contrast; flagged in output and paper).
  cov  = z(pre-belief) + z(word count); IC z-scored WITHIN arm.
  model= Gamma(log) GLM; arm x [z(IC)+z(IC)^2] interaction; LR + Wald.
  Positive control reproduced first: treatment |Delta| mean, control |Delta|
  mean, their ratio (~3.3x), and the treatment raw-scale |Delta| quadratic.

Output: prints; writes note09b_multiplicative_test_output.txt.
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
from scipy import stats as sp

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
spec06 = importlib.util.spec_from_file_location(
    "n09", HERE / "06_absolute_change_engagement.py")
n09 = importlib.util.module_from_spec(spec06)
spec06.loader.exec_module(n09)


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


def run():
    df = n09.load_costello()
    df = df.dropna(subset=["ic_q400", "abs_change", "Pre_Belief_Specific",
                           "OpenendedResponseWordCount", "is_treatment"]).copy()
    t = df[df.is_treatment == 1].copy()
    c = df[df.is_treatment == 0].copy()
    print(f"Treatment N = {len(t)}, Control N = {len(c)}")

    # ---- Positive control: reproduce published descriptives + treatment quad
    mt_, mc_ = t["abs_change"].mean(), c["abs_change"].mean()
    print(f"(0a) |Delta| mean: treatment = {mt_:.2f}, control = {mc_:.2f}, "
          f"ratio = {mt_/mc_:.2f}x  [published ~3.3x]")
    yt = t["abs_change"].values
    icz_t = zs(t["ic_q400"].values)
    Xt = sm.add_constant(np.column_stack([
        icz_t, icz_t**2, zs(t["Pre_Belief_Specific"].values),
        zs(t["OpenendedResponseWordCount"].values)]))
    mt = sm.OLS(yt, Xt).fit()
    # raw-scale quad for the -3.53 anchor
    icr = t["ic_q400"].values
    Xtr = sm.add_constant(np.column_stack([
        icr, icr**2, zs(t["Pre_Belief_Specific"].values),
        zs(t["OpenendedResponseWordCount"].values)]))
    mtr = sm.OLS(yt, Xtr).fit()
    print(f"(0b) treatment |Delta| quad: z(IC)^2 b = {mt.params[2]:+.2f} "
          f"(p = {mt.pvalues[2]:.4f})  [published -1.72]; "
          f"raw IC^2 b = {mtr.params[2]:+.2f}  [published -3.53]")
    cy = c["abs_change"].values
    icz_c = zs(c["ic_q400"].values)
    Xc = sm.add_constant(np.column_stack([
        icz_c, icz_c**2, zs(c["Pre_Belief_Specific"].values),
        zs(c["OpenendedResponseWordCount"].values)]))
    mc = sm.OLS(cy, Xc).fit()
    print(f"(0c) control   |Delta| quad: z(IC)^2 b = {mc.params[2]:+.2f} "
          f"(p = {mc.pvalues[2]:.4f}); raw-scale ratio treat/control = "
          f"{mtr.params[2] / sm.OLS(cy, sm.add_constant(np.column_stack([c['ic_q400'].values, c['ic_q400'].values**2, zs(c['Pre_Belief_Specific'].values), zs(c['OpenendedResponseWordCount'].values)]))).fit().params[2]:.2f}x")
    print()

    # ---- tab:absdv_extremity Costello rows (raw IC^2 scale, treatment arm).
    # M0: raw IC + raw IC^2 + z(pre) + z(wc); M1: + z|pre-50|; M2: + IC x z|pre-50|.
    abspre = np.abs(t["Pre_Belief_Specific"].values - 50.0)
    icr_t = t["ic_q400"].values
    prez, wcz = zs(t["Pre_Belief_Specific"].values), zs(t["OpenendedResponseWordCount"].values)
    apz = zs(abspre)
    m_m0 = sm.OLS(yt, sm.add_constant(np.column_stack([icr_t, icr_t**2, prez, wcz]))).fit()
    m_m1 = sm.OLS(yt, sm.add_constant(np.column_stack([icr_t, icr_t**2, prez, wcz, apz]))).fit()
    m_m2 = sm.OLS(yt, sm.add_constant(np.column_stack([icr_t, icr_t**2, prez, wcz, apz, icr_t*apz]))).fit()
    print("(0d) tab:absdv_extremity Costello (quad), raw IC^2 scale:")
    print(f"     M0 (IC+IC^2+controls)        : beta_IC^2 = {m_m0.params[2]:+.3f}, p = {m_m0.pvalues[2]:.4f}  [table -3.56]")
    print(f"     M1 (M0 + |pre-50|)           : beta_IC^2 = {m_m1.params[2]:+.3f}, p = {m_m1.pvalues[2]:.4f}  [table -3.44]")
    print(f"     M2 (M1 + IC x |pre-50|)      : beta_IC^2 = {m_m2.params[2]:+.3f}, p = {m_m2.pvalues[2]:.4f}; "
          f"interaction p = {m_m2.pvalues[6]:.4f}  [table -3.19, interaction p=.11]")
    print()

    # ---- The multiplicative test: Gamma(log) GLM, pooled, arm x [IC+IC^2]
    # IC z-scored within arm so the interaction is about SHAPE not level.
    t2 = pd.DataFrame({"y": t["abs_change"].values + 1.0, "icz": zs(t["ic_q400"].values),
                       "pre": zs(t["Pre_Belief_Specific"].values),
                       "wc": zs(t["OpenendedResponseWordCount"].values), "arm": 1.0})
    c2 = pd.DataFrame({"y": c["abs_change"].values + 1.0, "icz": zs(c["ic_q400"].values),
                       "pre": zs(c["Pre_Belief_Specific"].values),
                       "wc": zs(c["OpenendedResponseWordCount"].values), "arm": 0.0})
    d = pd.concat([t2, c2], ignore_index=True)
    base = np.column_stack([d.arm, d.icz, d.icz**2, d.pre, d.wc])
    inter = np.column_stack([d.arm * d.icz, d.arm * d.icz**2])
    X0 = sm.add_constant(base)
    X1 = sm.add_constant(np.column_stack([base, inter]))
    g0 = sm.GLM(d.y, X0, family=sm.families.Gamma(sm.families.links.log())).fit()
    g1 = sm.GLM(d.y, X1, family=sm.families.Gamma(sm.families.links.log())).fit()
    # LR via deviance/scale
    lr = (g0.deviance - g1.deviance) / g1.scale
    p_lr = sp.chi2.sf(lr, 2)
    b_int_quad = g1.params[-1]; p_int_quad = g1.pvalues[-1]
    print("(1) GAMMA(log) GLM on |Delta|+1, arm x [z(IC)+z(IC)^2] interaction:")
    print(f"    joint interaction LR(df=2) = {lr:.2f}, p = {p_lr:.3f}")
    print(f"    arm x z(IC)^2 (log scale): b = {b_int_quad:+.4f} "
          f"(p = {p_int_quad:.4f}); negative = treatment curvature exceeds "
          f"pure multiplicative scaling")
    # within-arm log-scale curvature for interpretation
    print(f"    [interpretation: if ~0, the {mt_/mc_:.1f}x raw-curvature ratio "
          f"is explained by the {mt_/mc_:.1f}x mean ratio alone]")
    print()

    # ---- Robustness: |Delta|>0 subset (no +1 offset)
    dpos = d[d.y > 1.0 + 1e-9].copy()  # |Delta|>0 i.e. y>1 after +1
    Xp0 = sm.add_constant(np.column_stack([dpos.arm, dpos.icz, dpos.icz**2,
                                           dpos.pre, dpos.wc]))
    Xp1 = sm.add_constant(np.column_stack([dpos.arm, dpos.icz, dpos.icz**2,
                                           dpos.pre, dpos.wc,
                                           dpos.arm*dpos.icz, dpos.arm*dpos.icz**2]))
    gp0 = sm.GLM(dpos.y - 1.0 + 1e-6, Xp0, family=sm.families.Gamma(sm.families.links.log())).fit()
    gp1 = sm.GLM(dpos.y - 1.0 + 1e-6, Xp1, family=sm.families.Gamma(sm.families.links.log())).fit()
    lrp = (gp0.deviance - gp1.deviance) / gp1.scale
    print(f"(2) robustness, |Delta|>0 subset (n = {len(dpos)}): "
          f"interaction LR(df=2) = {lrp:.2f}, p = {sp.chi2.sf(lrp,2):.3f}; "
          f"arm x z(IC)^2 b = {gp1.params[-1]:+.4f} (p = {gp1.pvalues[-1]:.4f})")
    print()
    print("CAVEAT (paper must state): treatment IC scored on pre-treatment text,")
    print("control IC on the post-dialogue conRestatement -- a scoring-window")
    print("asymmetry that this arm contrast cannot disentangle from a genuine")
    print("treatment effect on curvature shape.")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    (HERE / "note09b_multiplicative_test_output.txt").write_text(text)
    sys.stdout.write(f"\nWrote {HERE / 'note09b_multiplicative_test_output.txt'}\n")
