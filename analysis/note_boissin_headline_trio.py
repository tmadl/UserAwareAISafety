#!/usr/bin/env python3
"""note_boissin_headline_trio.py — consolidated reproduction of the three
Boissin headline numbers cited in the main text and SI Notes 9/12.

Main text / SI cite, on the validated Qwen3.5-ORPO-400 logit-EV scorer
(text_all = full-dialogue, Boissin primary IC source):

  (1) |Delta| monotone-descent leg (Note 9, Fig. tab:absdv / caption):
        beta_lin = -2.27 on |Delta|, p = .001, BF10 = 8.72
        AI x Human-like cell: beta = -3.47, p = .006
  (2) Raw signed-DV slope (weak opposite leg):
        beta_lin = +1.60, p = .035, BF10 = 0.30
  (3) Directional movers-toward-target (487 movers, signed Delta = post-pre,
      Boissin's original-study convention):
        beta_lin = +4.11, p < .001
      away-movers (234): beta_lin = -0.67, p = .22

Spec (paper convention for Boissin, see SI Note 12): IC and PreBelief on the
RAW scale; DV is belief_change (post - pre) or its absolute value. BF10 for the
linear IC term is BIC-based, comparing the model with IC against the same model
without IC (i.e. DV ~ PreBelief only). This is the BF reported in note12 /
note11 and matches the manuscript's BF definition for the linear leg.

All numbers recomputed on the length-bug-fixed boissin scores
(data/ic_qwen3orpo400/boissin_texts_for_scoring_qwenorpo400.jsonl, which is
the fixed rescoring; see analysis/_migrate_to_lengthfix.py).

Output: prints; writes note_boissin_headline_trio_output.txt.
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
    an = pd.read_csv(DATA / "boissin2025" / "analysis_data.csv")
    rows = [json.loads(l) for l in open(DATA_Q / "boissin_texts_for_scoring_qwenorpo400.jsonl")]
    ic = pd.DataFrame({
        "participantId": [r["participantId"] for r in rows],
        "IC":            [r["ic_qwenorpo400_all_logit"] for r in rows],
    })
    df = an.merge(ic, on="participantId", how="left")
    for c in ["belief_change", "PreBelief", "IC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["belief_change", "PreBelief", "IC"]).copy()
    df["abs_delta"] = df["belief_change"].abs()
    return df


def lin_with_bf(sub, dv):
    """DV ~ IC + PreBelief (raw scale). BF10(IC) = BIC(no-IC) vs BIC(with-IC)."""
    y = sub[dv].values.astype(float)
    ic = sub["IC"].values.astype(float)
    pre = sub["PreBelief"].values.astype(float)
    X_full = sm.add_constant(np.column_stack([ic, pre]))
    X_red = sm.add_constant(pre.reshape(-1, 1))
    m_full = sm.OLS(y, X_full).fit()
    m_red = sm.OLS(y, X_red).fit()
    bf = float(np.exp((m_red.bic - m_full.bic) / 2))
    return dict(n=len(sub), beta=m_full.params[1], p=m_full.pvalues[1], bf=bf)


def main():
    df = load_boissin()
    print("Boissin headline trio (Qwen3.5-ORPO-400 logit-EV, text_all, length-fixed)")
    print(f"N (complete cases) = {len(df)}\n")

    print("(1) |Delta| monotone-descent leg  [manuscript: -2.27, p=.001, BF10=8.72]")
    r = lin_with_bf(df, "abs_delta")
    print(f"    |Delta| ~ IC + PreBelief : beta_lin = {r['beta']:+.3f}, "
          f"p = {fmt_p(r['p'])}, BF10 = {r['bf']:.2f}, n = {r['n']}")
    aihl = df[(df["Speaker"] == "AI") & (df["PromptType"] == "Human-like")]
    ra = lin_with_bf(aihl, "abs_delta")
    print(f"    AI x Human-like cell     : beta_lin = {ra['beta']:+.3f}, "
          f"p = {fmt_p(ra['p'])}, BF10 = {ra['bf']:.2f}, n = {ra['n']}  "
          f"[manuscript: -3.47, p=.006]")

    print("\n(2) Raw signed-DV slope (weak opposite leg)  [manuscript: +1.60, p=.035, BF10=0.30]")
    r = lin_with_bf(df, "belief_change")
    print(f"    belief_change ~ IC + PreBelief : beta_lin = {r['beta']:+.3f}, "
          f"p = {fmt_p(r['p'])}, BF10 = {r['bf']:.2f}, n = {r['n']}")

    print("\n(3) Directional movers (signed Delta = post-pre)  [manuscript toward: +4.11, p<.001; away: -0.67, p=.22]")
    toward = df[df["belief_change"] < 0]   # moved toward debunking target
    away = df[df["belief_change"] > 0]      # moved away (backfired)
    rt = lin_with_bf(toward, "belief_change")
    rw = lin_with_bf(away, "belief_change")
    print(f"    toward-target (Delta<0): beta_lin = {rt['beta']:+.3f}, "
          f"p = {fmt_p(rt['p'])}, n = {rt['n']}")
    print(f"    away-target   (Delta>0): beta_lin = {rw['beta']:+.3f}, "
          f"p = {fmt_p(rw['p'])}, n = {rw['n']}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    out = buf.getvalue()
    print(out, end="")
    (Path(__file__).parent / "note_boissin_headline_trio_output.txt").write_text(out)
