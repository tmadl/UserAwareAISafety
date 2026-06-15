#!/usr/bin/env python3
"""note_discriminant_table.py — recompute every row of SI Table tab:discriminant
("Discriminant validity: IC predicts outcomes beyond surface text features") on
the length-bug-fixed Qwen3.5-ORPO-400 logit-EV scores.

Table structure (per row): covariate-only R2, +IC R2, DeltaR2, F, p
  where the IC block adds z(IC) + z(IC^2) (2 numerator df) over the
  covariate-only model, and F/p come from the nested-model incremental-F test.

Rows:
  - Costello : DV = DV_BeliefChange_Specific ; covs = z(wc) + z(pre) ; Q400 IC
               (length-fixed; consistent with the headline)
  - Boissin (pooled)   : DV = belief_change ; cov = z(PreBelief) ; Q400 IC_all
  - Boissin (AI x HL)  : same, subset Speaker==AI & PromptType==Human-like
  - Cheng S3 (gpt-4.1 baseline) : DV = rightorwrong ; cov = z(wc_all) ;
               gpt-4.1-mini IC_openai (UNCHANGED scorer family; for reference)

The Boissin (and Costello) rows are recomputed on length-fixed Q400 scores
because the recent length-fix pass had not yet propagated to this table.

Output: prints; writes note_discriminant_table_output.txt next to this script.
"""
import io
import json
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DCOST = DATA / "costello2024"
DQ400 = DATA / "ic_qwen3orpo400"


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / np.nanstd(v, ddof=1)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def incremental_F(y, X_cov, X_full):
    """Nested incremental-F test for adding the IC block.
    Returns (R2_cov, R2_full, dR2, F, p, df_num, df_den)."""
    m_cov = sm.OLS(y, X_cov).fit()
    m_full = sm.OLS(y, X_full).fit()
    df_num = X_full.shape[1] - X_cov.shape[1]
    df_den = int(m_full.df_resid)
    dr2 = m_full.rsquared - m_cov.rsquared
    F = (dr2 / df_num) / ((1 - m_full.rsquared) / df_den)
    p = sp.f.sf(F, df_num, df_den)
    return m_cov.rsquared, m_full.rsquared, dr2, F, p, df_num, df_den


def load_costello():
    an = pd.read_csv(DCOST / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DCOST / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DQ400 / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC_q400": pd.to_numeric(q["ic_qwenorpo400_logit"], errors="coerce").values,
    })
    df = an.merge(ic, on="participantId", how="inner")
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "IC_q400"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_boissin():
    an = pd.read_csv(DATA / "boissin2025" / "analysis_data.csv")
    rows = [json.loads(l) for l in
            open(DQ400 / "boissin_texts_for_scoring_qwenorpo400.jsonl")]
    ic = pd.DataFrame({
        "participantId": [r["participantId"] for r in rows],
        "IC_all": [r["ic_qwenorpo400_all_logit"] for r in rows],
    })
    df = an.merge(ic, on="participantId", how="left")
    for c in ["belief_change", "PreBelief", "IC_all"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_cheng_s3():
    an = pd.read_csv(DATA / "cheng2006" / "analysis_data.csv")
    cx = pd.read_csv(DATA / "cheng2006" / "all_complexity_scores.csv")[
        ["participantId", "IC_openai"]]
    df = an.merge(cx, on="participantId", how="inner")
    df = df[df["study"] == 3].copy()
    for c in ["rightorwrong", "wc_all", "IC_openai"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def row(label, y, cov_cols_z, ic_raw):
    """Build cov/full design and print one table row."""
    mask = np.isfinite(y) & np.isfinite(ic_raw)
    for c in cov_cols_z:
        mask &= np.isfinite(c)
    y2 = y[mask]
    covz = [zs(c[mask]) for c in cov_cols_z]
    icz = zs(ic_raw[mask])
    ic2z = zs(ic_raw[mask] ** 2)
    X_cov = sm.add_constant(np.column_stack(covz)) if covz else \
        np.ones((len(y2), 1))
    X_full = sm.add_constant(np.column_stack(covz + [icz, ic2z]))
    r2c, r2f, dr2, F, p, dfn, dfd = incremental_F(y2, X_cov, X_full)
    print(f"{label:<34s} n={len(y2):>4d}  cov R2={r2c:.3f}  +IC R2={r2f:.3f}  "
          f"dR2={dr2:.3f}  F({dfn},{dfd})={F:.2f}  p={fmt_p(p)}")
    return dict(n=len(y2), r2c=r2c, r2f=r2f, dr2=dr2, F=F, p=p)


def main():
    print("=" * 78)
    print("SI Table tab:discriminant — recompute on length-fixed Q400 IC")
    print("IC block = z(IC) + z(IC^2) (2 df) added over covariate-only model")
    print("=" * 78)

    # --- Costello (Q400, length-fixed) ---
    c = load_costello().dropna(
        subset=["DV_BeliefChange_Specific", "Pre_Belief_Specific",
                "OpenendedResponseWordCount", "IC_q400"])
    row("Costello (Specific belief change)",
        c["DV_BeliefChange_Specific"].values,
        [c["OpenendedResponseWordCount"].values, c["Pre_Belief_Specific"].values],
        c["IC_q400"].values)

    # --- Boissin pooled (Q400 IC_all, length-fixed) ---
    b = load_boissin().dropna(subset=["belief_change", "PreBelief", "IC_all"])
    row("Boissin (pooled)",
        b["belief_change"].values, [b["PreBelief"].values], b["IC_all"].values)

    # --- Boissin AI x Human-like cell ---
    bhl = b[(b["Speaker"] == "AI") & (b["PromptType"] == "Human-like")].copy()
    row("Boissin (AI x HL)",
        bhl["belief_change"].values, [bhl["PreBelief"].values],
        bhl["IC_all"].values)

    # --- Cheng S3 (gpt-4.1 baseline; scorer family UNCHANGED) ---
    ch = load_cheng_s3().dropna(subset=["rightorwrong", "wc_all", "IC_openai"])
    row("Cheng S3 (gpt-4.1 baseline)",
        ch["rightorwrong"].values, [ch["wc_all"].values], ch["IC_openai"].values)

    print("\nNote: Costello/Boissin rows on length-fixed Q400; Cheng row on the")
    print("gpt-4.1-mini baseline (IC_openai), scorer family unchanged.")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    (Path(__file__).resolve().parent /
     "note_discriminant_table_output.txt").write_text(text)
    import sys
    sys.stdout.write(text)
