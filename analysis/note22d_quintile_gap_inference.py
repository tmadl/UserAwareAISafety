#!/usr/bin/env python3
"""note22d_quintile_gap_inference.py — significance tests for the per-quintile
treatment-vs-control adverse-movement gaps (reviewer request: the published
"+3.4pp at Q1 / +2.6pp at Q5" tail-elevation claim carried no inference).

Builds the same arm-specific IC-quintile split as 11_baseline_anchor.py
(treatment IC from pre-treatment text, control IC from conRestatement;
quintiles within arm), then reports for each quintile:
  - two-proportion z-test (treatment vs control adverse rate, DV <= -5)
  - and a pooled logistic model: adverse ~ arm * quintile-indicator
Also reports the joint arm x quintile interaction (LR test, all quintiles).

Extended (resolution sweep): the same per-bin gap tests at halves, terciles,
quartiles, quintiles, and deciles; a planned 1-df tails-vs-middle contrast
(arm x is_tail logistic interaction, the McGuire-predicted pattern); and a
binning-free continuous test (arm x [z(IC) + z(IC)^2] logistic interaction,
IC z-scored within arm). The multi-resolution sweep is reported in full to
avoid resolution cherry-picking; the planned contrast and the continuous
interaction are the inferential tests, bins are description.

Output: prints; writes note22d_quintile_gap_inference_output.txt.
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
DATA = HERE.parent / "data" / "costello2024"
DATA_Q400 = HERE.parent / "data" / "ic_qwen3orpo400"


def load_arms():
    """Replicate 11_baseline_anchor.py's frame construction."""
    import json
    an = pd.read_csv(DATA / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q400_t = pd.read_csv(
        DATA_Q400 / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic_t = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC": q400_t["ic_qwenorpo400_logit"].astype(float).values})
    raw = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
    orig = pd.read_csv(raw, low_memory=False).drop_duplicates(
        subset="participantId", keep="first")
    ctrl = orig[orig["ExperimentalCondition"] == "Control"].copy()
    ic_c = pd.read_csv(DATA / "costello_controls_qwenorpo400.csv")[
        ["participantId", "ic_qwenorpo400_logit"]].rename(
        columns={"ic_qwenorpo400_logit": "IC"})
    ic_c["IC"] = pd.to_numeric(ic_c["IC"], errors="coerce")

    an["DV_BeliefChange_Specific"] = pd.to_numeric(
        an["DV_BeliefChange_Specific"], errors="coerce")
    ctrl["DV_BeliefChange_Specific"] = pd.to_numeric(
        ctrl["DV_BeliefChange_Specific"], errors="coerce")
    t = an.merge(ic_t, on="participantId", how="inner")
    c = ctrl.merge(ic_c, on="participantId", how="inner")
    for d in (t, c):
        d.dropna(subset=["IC", "DV_BeliefChange_Specific"], inplace=True)
    t = t.drop_duplicates("participantId").copy()
    c = c.drop_duplicates("participantId").copy()
    for d in (t, c):
        d["adverse"] = (d["DV_BeliefChange_Specific"] <= -5).astype(int)
        d["q"] = pd.qcut(d["IC"].rank(method="first"), 5, labels=False)
    return t, c


def zs_w(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * sp.norm.sf(abs(z))


def run():
    t, c = load_arms()
    print(f"Treatment N = {len(t)} (adverse {t['adverse'].mean():.1%}); "
          f"Control N = {len(c)} (adverse {c['adverse'].mean():.1%})")
    print()
    print("Per-quintile treatment-vs-control adverse-rate gaps "
          "(two-proportion z, two-tailed):")
    print(f"{'Q':>2s} {'n_t':>5s} {'rate_t':>7s} {'n_c':>5s} {'rate_c':>7s} "
          f"{'gap_pp':>7s} {'z':>6s} {'p':>7s}")
    for q in range(5):
        ts = t[t["q"] == q]; cs = c[c["q"] == q]
        x1, n1 = ts["adverse"].sum(), len(ts)
        x2, n2 = cs["adverse"].sum(), len(cs)
        z, p = two_prop_z(x1, n1, x2, n2)
        print(f"{q+1:>2d} {n1:>5d} {x1/n1:>7.1%} {n2:>5d} {x2/n2:>7.1%} "
              f"{(x1/n1 - x2/n2)*100:>+7.1f} {z:>6.2f} {p:>7.3f}")

    # Pooled logistic: adverse ~ arm * C(quintile), LR test for interaction
    t2 = t[["adverse", "q"]].copy(); t2["arm"] = 1
    c2 = c[["adverse", "q"]].copy(); c2["arm"] = 0
    d = pd.concat([t2, c2], ignore_index=True)
    qd = pd.get_dummies(d["q"], prefix="q", drop_first=True).astype(float)
    X0 = sm.add_constant(np.column_stack([d["arm"].values, qd.values]))
    inter = qd.values * d["arm"].values.reshape(-1, 1)
    X1 = sm.add_constant(np.column_stack([d["arm"].values, qd.values, inter]))
    m0 = sm.Logit(d["adverse"].values, X0).fit(disp=0)
    m1 = sm.Logit(d["adverse"].values, X1).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf)
    p_lr = sp.chi2.sf(lr, df=4)
    print()
    print(f"Joint arm x quintile interaction (logistic LR, df=4): "
          f"chi2 = {lr:.2f}, p = {p_lr:.3f}")

    # ---- Resolution sweep: halves / terciles / quartiles / quintiles / deciles
    print()
    print("=" * 72)
    print("RESOLUTION SWEEP (per-bin two-proportion z, two-tailed)")
    for k in (2, 3, 4, 5, 10):
        tt = t.copy(); cc = c.copy()
        tt["b"] = pd.qcut(tt["IC"].rank(method="first"), k, labels=False)
        cc["b"] = pd.qcut(cc["IC"].rank(method="first"), k, labels=False)
        print(f"\n-- k = {k} --")
        print(f"{'bin':>4s} {'n_t':>5s} {'rate_t':>7s} {'n_c':>5s} {'rate_c':>7s} "
              f"{'gap_pp':>7s} {'z':>6s} {'p':>7s}")
        for b in range(k):
            ts_ = tt[tt["b"] == b]; cs_ = cc[cc["b"] == b]
            x1, n1 = ts_["adverse"].sum(), len(ts_)
            x2, n2 = cs_["adverse"].sum(), len(cs_)
            z, pp = two_prop_z(x1, n1, x2, n2)
            print(f"{b+1:>4d} {n1:>5d} {x1/n1:>7.1%} {n2:>5d} {x2/n2:>7.1%} "
                  f"{(x1/n1 - x2/n2)*100:>+7.1f} {z:>6.2f} {pp:>7.3f}")

        if k >= 3:
            # planned 1-df contrast: tails (bottom+top bin) vs middle
            tt["tail"] = ((tt["b"] == 0) | (tt["b"] == k - 1)).astype(int)
            cc["tail"] = ((cc["b"] == 0) | (cc["b"] == k - 1)).astype(int)
            t3 = tt[["adverse", "tail"]].copy(); t3["arm"] = 1
            c3 = cc[["adverse", "tail"]].copy(); c3["arm"] = 0
            dd = pd.concat([t3, c3], ignore_index=True)
            Xt = sm.add_constant(np.column_stack([
                dd["arm"].values, dd["tail"].values,
                dd["arm"].values * dd["tail"].values]))
            mt = sm.Logit(dd["adverse"].values, Xt).fit(disp=0)
            print(f"  planned tails-vs-middle contrast (arm x tail): "
                  f"b = {mt.params[3]:+.3f}, z = {mt.tvalues[3]:.2f}, "
                  f"p = {mt.pvalues[3]:.3f}")

    # ---- Continuous (binning-free): arm x [z(IC) + z(IC)^2], z within arm
    print()
    print("=" * 72)
    print("CONTINUOUS TEST (no binning; IC z-scored within arm)")
    t4 = t[["adverse"]].copy(); t4["icz"] = zs_w(t["IC"].values); t4["arm"] = 1
    c4 = c[["adverse"]].copy(); c4["icz"] = zs_w(c["IC"].values); c4["arm"] = 0
    d4 = pd.concat([t4, c4], ignore_index=True)
    icz = d4["icz"].values; arm = d4["arm"].values
    X0c = sm.add_constant(np.column_stack([arm, icz, icz**2]))
    X1c = sm.add_constant(np.column_stack([arm, icz, icz**2,
                                           arm * icz, arm * icz**2]))
    m0c = sm.Logit(d4["adverse"].values, X0c).fit(disp=0)
    m1c = sm.Logit(d4["adverse"].values, X1c).fit(disp=0)
    lrc = 2 * (m1c.llf - m0c.llf)
    print(f"arm x [z(IC)+z(IC)^2] joint LR (df=2): chi2 = {lrc:.2f}, "
          f"p = {sp.chi2.sf(lrc, 2):.3f}")
    print(f"  arm x z(IC):   b = {m1c.params[4]:+.3f} (p = {m1c.pvalues[4]:.3f})")
    print(f"  arm x z(IC)^2: b = {m1c.params[5]:+.3f} (p = {m1c.pvalues[5]:.3f})")

    # ---- Same-frame sanity checks (DV comparison + covariate-adjusted) ----
    print()
    print("=" * 72)
    print("SAME-FRAME DV COMPARISON (why binary-adverse differs from belief-change)")
    # (i) Continuous-DV interaction in this exact frame: OLS belief change
    #     ~ arm * (z(IC) + z(IC)^2), IC z within arm. Should reproduce the
    #     published pooled interaction significance.
    t5 = t[["DV_BeliefChange_Specific", "adverse"]].copy()
    t5["icz"] = zs_w(t["IC"].values); t5["arm"] = 1
    t5["pre"] = pd.to_numeric(t["Pre_Belief_Specific"], errors="coerce")
    c5 = c[["DV_BeliefChange_Specific", "adverse"]].copy()
    c5["icz"] = zs_w(c["IC"].values); c5["arm"] = 0
    c5["pre"] = pd.to_numeric(c["Pre_Belief_Specific"], errors="coerce")
    d5 = pd.concat([t5, c5], ignore_index=True).dropna(subset=["pre"])
    d5["prez"] = zs_w(d5["pre"].values)
    icz = d5["icz"].values; arm = d5["arm"].values; prez = d5["prez"].values
    y_cont = d5["DV_BeliefChange_Specific"].values

    X0 = sm.add_constant(np.column_stack([arm, icz, icz**2, prez]))
    X1 = sm.add_constant(np.column_stack([arm, icz, icz**2, prez,
                                          arm*icz, arm*icz**2]))
    m0 = sm.OLS(y_cont, X0).fit()
    m1 = sm.OLS(y_cont, X1).fit()
    df_d = m1.df_model - m0.df_model
    f = ((m0.ssr - m1.ssr)/df_d) / (m1.ssr/m1.df_resid)
    p_f = sp.f.sf(f, df_d, m1.df_resid)
    print(f"(i) OLS belief change ~ arm x [z(IC)+z(IC)^2] + pre (N={len(d5)}):")
    print(f"    joint interaction F({int(df_d)},{int(m1.df_resid)}) = {f:.2f}, "
          f"p = {p_f:.5f}; arm x z(IC)^2 b = {m1.params[6]:+.2f} "
          f"(p = {m1.pvalues[6]:.4f})")

    # (ii) Binary adverse logistic WITH pre-belief covariate
    yb = d5["adverse"].values
    Xb0 = sm.add_constant(np.column_stack([arm, icz, icz**2, prez]))
    Xb1 = sm.add_constant(np.column_stack([arm, icz, icz**2, prez,
                                           arm*icz, arm*icz**2]))
    mb0 = sm.Logit(yb, Xb0).fit(disp=0)
    mb1 = sm.Logit(yb, Xb1).fit(disp=0)
    lrb = 2*(mb1.llf - mb0.llf)
    print(f"(ii) logistic adverse ~ arm x [z(IC)+z(IC)^2] + pre:")
    print(f"    joint LR (df=2) chi2 = {lrb:.2f}, p = {sp.chi2.sf(lrb,2):.3f}; "
          f"arm x z(IC)^2 b = {mb1.params[6]:+.3f} (p = {mb1.pvalues[6]:.3f})")

    # (iii) Linear probability model on adverse (matches percentage-point bins)
    ml = sm.OLS(yb, Xb1).fit(cov_type="HC1")
    ml0 = sm.OLS(yb, Xb0).fit(cov_type="HC1")
    fl = ((ml0.ssr - ml.ssr)/2) / (ml.ssr/ml.df_resid)
    print(f"(iii) LPM adverse (HC1): arm x z(IC)^2 b = {ml.params[6]:+.4f} "
          f"(p = {ml.pvalues[6]:.3f}); joint F approx p = {sp.f.sf(fl,2,ml.df_resid):.3f}")

    # U-shape test within treatment arm: adverse ~ z(IC) + z(IC)^2
    def zs(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / v.std(ddof=1)
    ic_z = zs(t["IC"].values)
    Xq = sm.add_constant(np.column_stack([ic_z, ic_z**2]))
    mq = sm.Logit(t["adverse"].values, Xq).fit(disp=0)
    print(f"Treatment-arm adverse ~ z(IC) + z(IC)^2 logistic: "
          f"b_lin = {mq.params[1]:+.3f} (p = {mq.pvalues[1]:.3f}), "
          f"b_quad = {mq.params[2]:+.3f} (p = {mq.pvalues[2]:.3f})")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    (HERE / "note22d_quintile_gap_inference_output.txt").write_text(text)
    sys.stdout.write(f"\nWrote {HERE / 'note22d_quintile_gap_inference_output.txt'}\n")
