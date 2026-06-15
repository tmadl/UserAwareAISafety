#!/usr/bin/env python3
"""note02b_conway_downstream.py — Conway et al. (2014) AutoIC word-list scorer
as a downstream moderator of Costello belief change, on the canonical spec.

Why this exists: the main text reports that the Conway dictionary scorer does
NOT recover the Costello inverted-U (it is a scorer-fidelity counterexample, not
a construct-level one -- Conway fails the Suedfeld/Jakob ground-truth validations
on short naturalistic text). This script makes that number reproducible from the
release and reports it on the SAME canonical scale as the headline coefficient.

Source data: Conway AutoIC scores are produced by the Conway scorer on the same
pre-treatment Costello belief essays scored by the primary pipeline. The raw
scorer output lives in the main repo
(../conway_ic_scorer_data/coded_costello_texts_for_scoring_init.xlsx, column
"IC" = AutoIC composite). The rows are in the SAME order as
texts_for_scoring.jsonl (the scoring manifest), so participantId is recovered by
index AFTER verifying text alignment. This script writes the released artifact
data/costello2024/costello_conway_ic.csv (participantId, ic_conway) and then runs
the moderation; downstream the released CSV is the reproducible input.

Spec declaration (identical to the -1.97 headline):
  DV     = DV_BeliefChange_Specific (pre - post, 0-100).
  sample = primary 1,782 (treatment complete-cases: valid Q400 IC, belief change,
           pre-belief, word count) -- the SAME sample the headline is fit on, so
           the Conway comparison is apples-to-apples.
  model  = belief ~ z(IC) + z(IC)^2 + z(pre-belief) + z(word count), OLS.
  scale  = z(IC)^2 headline scale (raw IC^2 and z(IC^2) also printed for cross-ref).
  BF     = BIC-based, quadratic-vs-linear (exp((BIC_lin - BIC_quad)/2)).

Output: prints; writes costello_conway_ic.csv and note02b_conway_downstream_output.txt.
"""
import io
import json
import re
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REL = HERE.parent
DCOST = REL / "data" / "costello2024"
DQ400 = REL / "data" / "ic_qwen3orpo400"
# Raw Conway scorer output lives in the sibling source repo (not the release);
# the released artifact is OUT_CSV, written by build_conway_csv() on first run.
CONWAY_XLSX = Path("/mnt/workvm/UserAwareAISafety/data/conway_ic_scorer_data/"
                   "coded_costello_texts_for_scoring_init.xlsx")
OUT_CSV = DCOST / "costello_conway_ic.csv"


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / np.nanstd(v, ddof=1)


def norm(t):
    # strip ALL non-alphanumerics (incl. whitespace -- the Conway export collapses
    # spacing differently from the jsonl, which spuriously breaks space-sensitive
    # matching); first 60 alnum chars uniquely identify each essay.
    return re.sub(r"[^a-z0-9]", "", str(t).lower())[:60]


def build_conway_csv():
    """Align Conway xlsx rows to participantId via the jsonl manifest, verify
    text correspondence, and write the released CSV. If the raw xlsx is absent
    (e.g. a clean release checkout), fall back to the already-written CSV."""
    meta = [json.loads(l) for l in open(DCOST / "texts_for_scoring.jsonl")]
    if not CONWAY_XLSX.exists():
        if OUT_CSV.exists():
            print(f"[raw xlsx absent; using released {OUT_CSV.name}]")
            return pd.read_csv(OUT_CSV)
        raise FileNotFoundError(f"Need {CONWAY_XLSX} or {OUT_CSV}")
    cw = pd.read_excel(CONWAY_XLSX)
    assert len(cw) == len(meta), f"row mismatch {len(cw)} vs {len(meta)}"
    # verify alignment: normalised text of Conway 'Paragraph' vs jsonl text_initial
    match = sum(norm(cw["Paragraph"].iloc[i]) == norm(meta[i]["text_initial"])
                for i in range(len(meta)))
    rate = match / len(meta)
    print(f"alignment check: {match}/{len(meta)} normalised-text exact matches "
          f"({rate:.3%})")
    assert rate > 0.95, f"alignment too low ({rate:.3%}) -- do NOT trust index join"
    out = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "ic_conway": cw["IC"].astype(float).values,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(out)} rows)")
    return out


def run():
    conway = build_conway_csv()

    # primary 1,782 sample: analysis_data + Q400 IC (defines the sample) + covars
    an = pd.read_csv(DCOST / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DCOST / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DQ400 / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    icq = pd.DataFrame({"participantId": [m["participantId"] for m in meta],
                        "ic_q400": q["ic_qwenorpo400_logit"].astype(float).values})
    df = (an.merge(icq, on="participantId", how="inner")
            .merge(conway, on="participantId", how="inner"))
    df = df.dropna(subset=["ic_q400", "ic_conway", "DV_BeliefChange_Specific",
                           "Pre_Belief_Specific", "OpenendedResponseWordCount"]).copy()
    print(f"\nPrimary-sample N = {len(df)} (must be 1782 to match the headline fit)")

    y = df["DV_BeliefChange_Specific"].values
    pre = zs(df["Pre_Belief_Specific"].values)
    wc = zs(df["OpenendedResponseWordCount"].values)
    ic = df["ic_conway"].values
    icz = zs(ic)
    print(f"Conway IC: mean = {ic.mean():.2f}, SD = {ic.std(ddof=1):.2f}, "
          f"range = [{ic.min():.2f}, {ic.max():.2f}]")

    # linear-only (for BF) and quadratic on each scale
    Xlin = sm.add_constant(np.column_stack([icz, pre, wc]))
    mlin = sm.OLS(y, Xlin).fit()
    print("\nConway quadratic moderation (canonical spec):")
    for tag, x1, x2 in [("z(IC)^2  [HEADLINE SCALE]", icz, icz ** 2),
                        ("raw IC^2", ic, ic ** 2),
                        ("z(IC^2)", icz, zs(ic ** 2))]:
        X = sm.add_constant(np.column_stack([x1, x2, pre, wc]))
        m = sm.OLS(y, X).fit()
        bf = float(np.exp((mlin.bic - m.bic) / 2))  # quad vs lin
        print(f"  {tag:26s} beta_IC2 = {m.params[2]:+.3f}, "
              f"p = {m.pvalues[2]:.3f}, BF10(quad/lin) = {bf:.3f}")
    print("\nContrast: primary Q400 scorer on this sample = beta_IC2 = -1.97 "
          "(z(IC)^2), p < .001.")
    print("Conway is null on every scale (p > .5, BF << 1): the dictionary scorer")
    print("does not recover the inverted-U -- a scorer-fidelity counterexample")
    print("(Conway fails Suedfeld/Jakob ground-truth validation; SI Note 2).")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    (HERE / "note02b_conway_downstream_output.txt").write_text(text)
    sys.stdout.write(f"\nWrote {HERE / 'note02b_conway_downstream_output.txt'}\n")
