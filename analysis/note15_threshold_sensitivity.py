#!/usr/bin/env python3
"""note15_threshold_sensitivity.py — SI Note 15.

Adverse-movement threshold sensitivity for the bottom-IC-quintile screening
claim. Varies the adverse-movement threshold from <0 through ≤-20 and shows
that the screen retains ~86% of large-changers and excludes 22-31% of adverse
movers across thresholds — adverse exclusion is monotonically larger at
stricter thresholds, consistent with low IC marking a reception-failure
regime rather than measurement-noise drift.
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"


def main():
    an = pd.read_csv(DATA / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC": q["ic_qwenorpo400_logit"].astype(float).values,
    })
    df = an.merge(ic, on="participantId", how="left")
    df["DV_BeliefChange_Specific"] = pd.to_numeric(df["DV_BeliefChange_Specific"], errors="coerce")
    df = df.dropna(subset=["DV_BeliefChange_Specific", "IC"]).copy()

    # Bottom IC quintile (Q1)
    df["ic_q"] = pd.qcut(df["IC"], 5, labels=False)
    bottom_q = df["ic_q"] == 0

    # Large-change definition: DV >= +20 (paper's headline threshold)
    LC = df["DV_BeliefChange_Specific"] >= 20

    print("SI Note 15 — Adverse-movement threshold sensitivity for the enrichment claim")
    print(f"Sample: Costello primary (n = {len(df)}), bottom IC quintile (Q1) = {bottom_q.sum()} participants\n")
    print(f"{'Adverse threshold':<25s} {'Total LC':>9s} {'Total Adv':>10s} {'LC retained':>12s} {'%':>5s} {'Adv excluded':>13s} {'%':>5s}")
    print("-" * 90)

    for label, adv_mask in [
        ("DV < 0",          df["DV_BeliefChange_Specific"] < 0),
        ("DV ≤ -5 (default)",  df["DV_BeliefChange_Specific"] <= -5),
        ("DV ≤ -10",        df["DV_BeliefChange_Specific"] <= -10),
        ("DV ≤ -15",        df["DV_BeliefChange_Specific"] <= -15),
        ("DV ≤ -20",        df["DV_BeliefChange_Specific"] <= -20),
    ]:
        # Drop bottom quintile: kept = ~bottom_q
        kept = ~bottom_q
        total_LC = LC.sum()
        total_Adv = adv_mask.sum()
        LC_retained = (LC & kept).sum()
        Adv_excluded = (adv_mask & bottom_q).sum()
        print(f"{label:<25s} {total_LC:>9d} {total_Adv:>10d} {LC_retained:>12d} "
              f"{100*LC_retained/total_LC:>4.0f}% {Adv_excluded:>13d} "
              f"{100*Adv_excluded/total_Adv:>4.0f}%")


if __name__ == "__main__":
    main()
