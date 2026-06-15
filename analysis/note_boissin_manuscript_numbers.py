#!/usr/bin/env python3
"""note_boissin_manuscript_numbers.py — recompute EVERY Boissin number cited in
the PNAS Nexus SI (si.tex), on the length-bug-fixed Qwen3.5-ORPO-400 logit-EV
scores.

This extends analysis/note_boissin_headline_trio.py: it reproduces the four
verified ANCHOR numbers (|Delta| full, signed full, directional movers,
AI x Human-like |Delta|) and then computes every remaining si.tex cell:

  STEP-1 anchors (must match):
    |Delta| full (n955):   beta_lin = -2.31, p<.001, BF(lin/null) = 11.47
    signed full (n955):    beta_lin = +1.72, p=.023, BF = 0.44
    movers toward Delta<0 (n487): +4.16, p<.001 ; away Delta>0 (n234): -0.59, p=.28
    AI x Human-like |Delta| (n244): -3.48, p=.005, BF = 3.25

  STEP-2 sites recomputed:
    (a) tab:absdv_summary  — 3 Boissin rows (beta_lin, beta_quad, p, R2=DeltaR2, BF)
    (b) tab:absdv_extremity — Boissin (lin) M0 / M1 + AI x HL extremity prose
    (c) tab:boissin_movers — Full/|D|>0/|D|>=5/toward/backfired (n, beta_lin, p)
    (d) two-source table — text_all (full) vs text_initial (pre-only),
        cols beta_lin, p, BF(lin/null), BF(quad/lin); + AI x HL pre vs full prose; r
    (e) tab:boissin_cells — 11 per-cell rows (n, beta_lin, p, BF lin vs null)
    (f) movers-toward ratio (toward-slope vs full-signed-slope) for line-567 prose

Spec (paper convention for Boissin, see SI Note 12):
  - DV = belief_change (post - pre) or |belief_change|.
  - Covariate = PreBelief ONLY (raw scale). IC raw scale.
  - BF(lin/null)  = BIC(PreBelief-only) vs BIC(IC + PreBelief).
  - BF(quad/lin)  = BIC(IC + PreBelief) vs BIC(IC + IC^2 + PreBelief).
  - beta_quad     = IC^2 coefficient from IC + IC^2 + PreBelief, raw IC scale.
  - "R2"/DeltaR2  = R2(IC + PreBelief) - R2(PreBelief-only).

Output: prints; writes note_boissin_manuscript_numbers_output.txt.
"""
import io
import json
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_Q = DATA / "ic_qwen3orpo400"


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def load_boissin():
    """Boissin analysis frame with both text_all and text_initial IC sources."""
    an = pd.read_csv(DATA / "boissin2025" / "analysis_data.csv")
    rows = [json.loads(l) for l in open(DATA_Q / "boissin_texts_for_scoring_qwenorpo400.jsonl")]
    ic = pd.DataFrame({
        "participantId": [r["participantId"] for r in rows],
        "IC_all":        [r["ic_qwenorpo400_all_logit"] for r in rows],
        "IC_init":       [r["ic_qwenorpo400_initial_logit"] for r in rows],
    })
    df = an.merge(ic, on="participantId", how="left")
    for c in ["belief_change", "PreBelief", "IC_all", "IC_init"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["belief_change", "PreBelief", "IC_all"]).copy()
    df["abs_delta"] = df["belief_change"].abs()
    df["pre_extr"] = (df["PreBelief"] - 50).abs()
    return df


def _r2(y, X):
    return sm.OLS(y, X).fit().rsquared


def lin_with_bf(sub, dv, iccol="IC_all"):
    """DV ~ IC + PreBelief (raw scale).
    BF(lin/null) = BIC(PreBelief-only) vs BIC(IC + PreBelief).
    Also returns beta_quad (IC^2 from IC+IC^2+PreBelief), BF(quad/lin),
    and DeltaR2 = R2(IC+pre) - R2(pre)."""
    y = sub[dv].values.astype(float)
    ic = sub[iccol].values.astype(float)
    pre = sub["PreBelief"].values.astype(float)

    X_pre = sm.add_constant(pre.reshape(-1, 1))
    X_lin = sm.add_constant(np.column_stack([ic, pre]))
    X_quad = sm.add_constant(np.column_stack([ic, ic ** 2, pre]))

    m_pre = sm.OLS(y, X_pre).fit()
    m_lin = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()

    bf_lin_null = float(np.exp((m_pre.bic - m_lin.bic) / 2))
    bf_quad_lin = float(np.exp((m_lin.bic - m_quad.bic) / 2))
    dr2 = m_lin.rsquared - m_pre.rsquared
    return dict(
        n=len(sub),
        beta=m_lin.params[1], p=m_lin.pvalues[1],
        beta_quad=m_quad.params[2], p_quad=m_quad.pvalues[2],
        bf=bf_lin_null, bf_quad_lin=bf_quad_lin, dr2=dr2,
    )


def extremity_models(sub, dv="abs_delta", iccol="IC_all"):
    """M0: IC + pre. M1: M0 + |pre-50|. M2: M1 + IC x |pre-50|.
    Returns IC linear coef/p for M0, M1, and interaction p for M2."""
    y = sub[dv].values.astype(float)
    ic = sub[iccol].values.astype(float)
    pre = sub["PreBelief"].values.astype(float)
    extr = sub["pre_extr"].values.astype(float)

    X0 = sm.add_constant(np.column_stack([ic, pre]))
    X1 = sm.add_constant(np.column_stack([ic, pre, extr]))
    X2 = sm.add_constant(np.column_stack([ic, pre, extr, ic * extr]))
    m0 = sm.OLS(y, X0).fit()
    m1 = sm.OLS(y, X1).fit()
    m2 = sm.OLS(y, X2).fit()
    return dict(
        m0_beta=m0.params[1], m0_p=m0.pvalues[1],
        m1_beta=m1.params[1], m1_p=m1.pvalues[1],
        m2_beta=m2.params[1], m2_p=m2.pvalues[1],
        m2_inter_beta=m2.params[4], m2_inter_p=m2.pvalues[4],
    )


def main():
    df = load_boissin()
    print("=" * 78)
    print("Boissin manuscript numbers (Qwen3.5-ORPO-400 logit-EV, length-fixed)")
    print(f"N (complete cases) = {len(df)}")
    print("=" * 78)

    # ---------------------------------------------------------------- ANCHORS
    print("\n### STEP 1 — ANCHORS (must match verified values) ###")
    r_abs = lin_with_bf(df, "abs_delta")
    print(f"|Delta| full (n{r_abs['n']}): beta_lin={r_abs['beta']:+.2f}, "
          f"p={fmt_p(r_abs['p'])}, BF(lin/null)={r_abs['bf']:.2f}  "
          f"[anchor: -2.31, p<.001, 11.47]")
    r_sgn = lin_with_bf(df, "belief_change")
    print(f"signed full  (n{r_sgn['n']}): beta_lin={r_sgn['beta']:+.2f}, "
          f"p={fmt_p(r_sgn['p'])}, BF={r_sgn['bf']:.2f}  "
          f"[anchor: +1.72, p=.023, 0.44]")
    tw = lin_with_bf(df[df["belief_change"] < 0], "belief_change")
    aw = lin_with_bf(df[df["belief_change"] > 0], "belief_change")
    print(f"toward Delta<0 (n{tw['n']}): {tw['beta']:+.2f}, p={fmt_p(tw['p'])}  "
          f"[anchor: +4.16, p<.001]")
    print(f"away   Delta>0 (n{aw['n']}): {aw['beta']:+.2f}, p={fmt_p(aw['p'])}  "
          f"[anchor: -0.59, p=.28]")
    aihl = df[(df["Speaker"] == "AI") & (df["PromptType"] == "Human-like")]
    r_ai = lin_with_bf(aihl, "abs_delta")
    print(f"AI x HL |Delta| (n{r_ai['n']}): {r_ai['beta']:+.2f}, "
          f"p={fmt_p(r_ai['p'])}, BF={r_ai['bf']:.2f}  "
          f"[anchor: -3.48, p=.005, 3.25]")

    # ----------------------------------------------- (a) tab:absdv_summary
    print("\n### (a) tab:absdv_summary — 3 Boissin rows "
          "[beta_lin, beta_quad, p, R2(=DeltaR2), BF] ###")
    print(f"Boissin (955) signed : beta_lin={r_sgn['beta']:+.2f}, "
          f"beta_quad={r_sgn['beta_quad']:+.2f}, p={r_sgn['p']:.3f} (lin), "
          f"R2={r_sgn['dr2']:.3f}, BF={r_sgn['bf']:.2f}")
    print(f"Boissin (955) |Delta|: beta_lin={r_abs['beta']:+.2f}, "
          f"beta_quad={r_abs['beta_quad']:+.2f}, p={r_abs['p']:.3f} (lin), "
          f"R2={r_abs['dr2']:.3f}, BF={r_abs['bf']:.2f}")
    print(f"Boissin AIxHL (244) |Delta|: beta_lin={r_ai['beta']:+.2f}, "
          f"beta_quad={r_ai['beta_quad']:+.2f}, p={r_ai['p']:.3f} (lin), "
          f"R2={r_ai['dr2']:.3f}, BF={r_ai['bf']:.2f}")

    # ----------------------------------------------- (b) tab:absdv_extremity
    print("\n### (b) tab:absdv_extremity — Boissin (lin) M0/M1/M2 ###")
    ex = extremity_models(df, "abs_delta")
    print(f"M0 (IC+pre)           : beta_IC={ex['m0_beta']:+.2f}, p={ex['m0_p']:.3f}")
    print(f"M1 (M0 + |pre-50|)    : beta_IC={ex['m1_beta']:+.2f}, p={ex['m1_p']:.3f}")
    print(f"M2 (M1 + IC x|pre-50|): beta_IC={ex['m2_beta']:+.2f}, p={ex['m2_p']:.3f}; "
          f"interaction beta={ex['m2_inter_beta']:+.4f}, p={ex['m2_inter_p']:.3f}")
    exa = extremity_models(aihl, "abs_delta")
    print(f"AI x HL extremity-adj : M1 beta_IC={exa['m1_beta']:+.2f}, "
          f"p={exa['m1_p']:.3f}; interaction p={exa['m2_inter_p']:.3f}")
    print("  prose@426 should read M1 (after |pre-50| adjustment).")

    # ----------------------------------------------- (c) tab:boissin_movers
    print("\n### (c) tab:boissin_movers — signed belief_change ~ IC + pre ###")
    subsets = [
        ("Full", df),
        ("|Delta|>0", df[df["abs_delta"] > 0]),
        ("|Delta|>=5", df[df["abs_delta"] >= 5]),
        ("Delta<0 toward", df[df["belief_change"] < 0]),
        ("Delta>0 backfired", df[df["belief_change"] > 0]),
    ]
    movers = {}
    for name, sub in subsets:
        rr = lin_with_bf(sub, "belief_change")
        movers[name] = rr
        print(f"{name:20s}: n={rr['n']:4d}  beta_lin={rr['beta']:+.2f}  "
              f"p={fmt_p(rr['p'])}")

    # ----------------------------------------------- (d) two-source table
    print("\n### (d) two-source table — text_all vs text_initial ###")
    full_all = lin_with_bf(df, "belief_change", iccol="IC_all")
    full_init = lin_with_bf(df, "belief_change", iccol="IC_init")
    print(f"Q400 text_all (full)    : beta_lin={full_all['beta']:+.2f}, "
          f"p={full_all['p']:.3f}, BF(lin/null)={full_all['bf']:.2f}, "
          f"BF(quad/lin)={full_all['bf_quad_lin']:.2f}")
    print(f"Q400 text_initial (pre) : beta_lin={full_init['beta']:+.2f}, "
          f"p={full_init['p']:.3f}, BF(lin/null)={full_init['bf']:.2f}, "
          f"BF(quad/lin)={full_init['bf_quad_lin']:.2f}")
    ai_all = lin_with_bf(aihl, "belief_change", iccol="IC_all")
    ai_init = lin_with_bf(aihl, "belief_change", iccol="IC_init")
    print(f"AI x HL prose@642: pre-only beta_lin={ai_init['beta']:+.2f} "
          f"(p={ai_init['p']:.3f}) vs full beta_lin={ai_all['beta']:+.2f} "
          f"(p={ai_all['p']:.3f})")
    r_src = np.corrcoef(df["IC_all"], df["IC_init"])[0, 1]
    print(f"r(IC_all, IC_init) on same participants = {r_src:.3f}")

    # ----------------------------------------------- (e) tab:boissin_cells
    print("\n### (e) tab:boissin_cells — per-cell signed fits "
          "[n, beta_lin, p, BF(lin vs null)] ###")
    EU = "Epistemically unwarranted beliefs"
    CO = "Conspiracy beliefs"
    cells = [
        ("Pooled", None, None, None),
        ("AI", "AI", None, None),
        ("AI Human-like", "AI", "Human-like", None),
        ("AI Human-like x Conspiracy", "AI", "Human-like", CO),
        ("AI Human-like x Epist.unw.", "AI", "Human-like", EU),
        ("AI Neutral x Conspiracy", "AI", "Neutral", CO),
        ("AI Neutral x Epist.", "AI", "Neutral", EU),
        ("Expert (all)", "Expert", None, None),
        ("Expert Human-like x Conspiracy", "Expert", "Human-like", CO),
        ("Expert Human-like x Epist.", "Expert", "Human-like", EU),
        ("Expert Neutral x Conspiracy", "Expert", "Neutral", CO),
        ("Expert Neutral x Epist.", "Expert", "Neutral", EU),
    ]
    for name, sp, pt, bt in cells:
        sub = df
        if sp is not None:
            sub = sub[sub["Speaker"] == sp]
        if pt is not None:
            sub = sub[sub["PromptType"] == pt]
        if bt is not None:
            sub = sub[sub["BeliefType"] == bt]
        rr = lin_with_bf(sub, "belief_change")
        print(f"{name:34s}: n={rr['n']:4d}  beta_lin={rr['beta']:+.2f}  "
              f"p={rr['p']:.3f}  BF={rr['bf']:.2f}")

    # ----------------------------------------------- (f) movers-toward ratio
    print("\n### (f) movers-toward ratio (line-567 prose) ###")
    ratio = movers["Delta<0 toward"]["beta"] / r_sgn["beta"]
    print(f"toward-slope (+{movers['Delta<0 toward']['beta']:.2f}) / "
          f"full-signed-slope (+{r_sgn['beta']:.2f}) = {ratio:.2f}x")
    print("  ('triples' is wrong; report ~2.4-fold)")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    out = buf.getvalue()
    print(out, end="")
    (Path(__file__).parent / "note_boissin_manuscript_numbers_output.txt").write_text(out)
