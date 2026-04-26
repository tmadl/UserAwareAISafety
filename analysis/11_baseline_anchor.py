#!/usr/bin/env python3
"""
11_baseline_anchor.py -- control-arm baseline anchoring for the 5-point
adverse-movement threshold.

Addresses the reviewer concern (autoreview3.md, Overall "Safety layer
operationalization") that 5-point adverse shifts in the treatment arm must
be demonstrated to exceed control-arm test-retest baseline noise before the
bottom-quintile flag can be cited as a safety enrichment instrument.

Computes:
  - control-arm >=5-point adverse-movement rate (overall and by IC quintile)
  - treatment-arm >=5-point adverse-movement rate (overall and by IC quintile)
  - treatment-minus-control gap at each quintile (treatment-attributable
    adverse-movement rate above the control baseline)
  - |DV| distribution in controls as a noise-floor check

DV convention: DV_BeliefChange_Specific = pre - post; adverse (belief
strengthening) = post > pre = DV <= -5.

Inputs (relative to repo root):
  - data/costello2024/analysis_data.csv
  - data/costello2024/texts_for_scoring.jsonl
  - data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv
  - data/costello2024/Data 8.28.24/AllDataForPublication.PPI.8.28.24.csv
  - data/costello2024/costello_controls_qwenorpo400.csv
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q400 = ROOT / "data" / "ic_qwen3orpo400"


# -- Treatment arm --
an = pd.read_csv(DATA / "analysis_data.csv")
meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
q400_t = pd.read_csv(DATA_Q400 / "costello_texts_for_scoring_initial_qwenorpo400.csv")
ic_q400_t = pd.DataFrame({
    "participantId": [m_["participantId"] for m_ in meta],
    "ic_q400": q400_t["ic_qwenorpo400_logit"].astype(float).values,
})
t = an.merge(ic_q400_t, on="participantId", how="inner")
t["DV_BeliefChange_Specific"] = pd.to_numeric(t["DV_BeliefChange_Specific"], errors="coerce")
t = t.dropna(subset=["DV_BeliefChange_Specific", "ic_q400"])

# -- Control arm --
from _raw_data_check import require_raw
_raw = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
require_raw(_raw, "Costello", "https://osf.io/gdkb7/")
orig = pd.read_csv(_raw, low_memory=False)
orig = orig.drop_duplicates(subset="participantId", keep="first")
ctrl = orig[orig["ExperimentalCondition"] == "Control"].copy()
ic_q400_c = pd.read_csv(DATA / "costello_controls_qwenorpo400.csv")[
    ["participantId", "ic_qwenorpo400_logit"]].rename(
    columns={"ic_qwenorpo400_logit": "ic_q400"})
c = ctrl.merge(ic_q400_c, on="participantId", how="inner")
c["DV_BeliefChange_Specific"] = pd.to_numeric(c["DV_BeliefChange_Specific"], errors="coerce")
c = c.dropna(subset=["DV_BeliefChange_Specific", "ic_q400"])


def adverse_rate(df):
    return (df["DV_BeliefChange_Specific"] <= -5).mean()


def favourable5_rate(df):
    return (df["DV_BeliefChange_Specific"] >= 5).mean()


def large_rate(df, thr=20):
    return (df["DV_BeliefChange_Specific"] >= thr).mean()


def by_quintile(df, label):
    df = df.copy()
    df["ic_q"] = pd.qcut(df["ic_q400"], 5, labels=[1, 2, 3, 4, 5])
    rows = []
    for q in [1, 2, 3, 4, 5]:
        sub = df[df["ic_q"] == q]
        rows.append({
            "arm": label, "quintile": q, "n": len(sub),
            "adverse_rate": adverse_rate(sub),
            "large_change_rate": large_rate(sub),
            "favourable5_rate": favourable5_rate(sub),
            "mean_dv": sub["DV_BeliefChange_Specific"].mean(),
        })
    return pd.DataFrame(rows)


print("=" * 85)
print("OVERALL -- >=5-point adverse-movement rate (DV <= -5)")
print("=" * 85)
print(f"Treatment: N = {len(t):5d}  adverse = {adverse_rate(t):6.1%}  "
      f"mean DV = {t['DV_BeliefChange_Specific'].mean():+.2f}")
print(f"Control:   N = {len(c):5d}  adverse = {adverse_rate(c):6.1%}  "
      f"mean DV = {c['DV_BeliefChange_Specific'].mean():+.2f}")
print(f"\nTreatment - Control adverse-rate gap: "
      f"{adverse_rate(t) - adverse_rate(c):+.1%} (abs) / "
      f"{adverse_rate(t) / adverse_rate(c):.2f}x (ratio)")

print("\n" + "=" * 85)
print("BY IC QUINTILE (within-arm quintile split)")
print("=" * 85)
t_q = by_quintile(t, "Treatment")
c_q = by_quintile(c, "Control")
print("\nTreatment arm:")
print(t_q.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nControl arm:")
print(c_q.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n" + "=" * 85)
print("TREATMENT-ATTRIBUTABLE ADVERSE RATE (T - C within each quintile)")
print("=" * 85)
for q in [1, 2, 3, 4, 5]:
    t_rate = t_q[t_q["quintile"] == q]["adverse_rate"].iloc[0]
    c_rate = c_q[c_q["quintile"] == q]["adverse_rate"].iloc[0]
    gap_pp = (t_rate - c_rate) * 100
    ratio = t_rate / c_rate if c_rate > 0 else float("nan")
    print(f"  Q{q}: T = {t_rate:5.1%}  C = {c_rate:5.1%}  gap = {gap_pp:+4.1f}pp  ratio = {ratio:.2f}x")

print("\n" + "=" * 85)
print("CONTROL |DV| DISTRIBUTION (noise floor)")
print("=" * 85)
c["abs_dv"] = c["DV_BeliefChange_Specific"].abs()
for thr in [0, 1, 3, 5, 10, 20]:
    print(f"  |DV| >= {thr:2d}: {(c['abs_dv'] >= thr).mean():6.1%}  (N = {len(c)})")

print(f"\nControl favourable >=5: {favourable5_rate(c):6.1%}")
print(f"Control adverse    >=5: {adverse_rate(c):6.1%}")
print(f"Ratio fav/adv: {favourable5_rate(c)/adverse_rate(c):.2f}  "
      f"(>1 => mild net-debunking drift even in controls)")
