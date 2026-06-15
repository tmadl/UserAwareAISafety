#!/usr/bin/env python3
"""note09_control_n_audit.py — reproduce the SI Note 9 Costello control-arm Ns
on the length-bug-fixed IC scores.

The SI Note 9 control-arm placebo paragraph cites three control sample sizes:
  - full active control arm                                   (manuscript: n = 1,152)
  - control rows merged with a per-participant IC score       (manuscript: 1,107)
  - + post-dialogue belief re-measurement (canonical spec)    (manuscript: 1,075)

The 1,107 / 1,075 figures predate the length-fix migration. On the fixed control
IC scores (data/costello2024/costello_controls_qwenorpo400.csv, the direct copy
of the length-fixed control rescoring) the merged / analyzable Ns move up. This
script prints the literal current values so the prose can be updated.
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DC = ROOT / "data" / "costello2024"
RAW = DC / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"


def run():
    icc = pd.read_csv(DC / "costello_controls_qwenorpo400.csv")
    orig = pd.read_csv(RAW, low_memory=False).drop_duplicates("participantId", keep="first")
    ctrl = orig[orig["ExperimentalCondition"] == "Control"].copy()

    full = len(icc)                       # full control IC-scored set
    merged = ctrl.merge(icc[["participantId", "ic_qwenorpo400_logit"]],
                        on="participantId", how="inner")
    n_merged = merged["ic_qwenorpo400_logit"].notna().sum()
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific", "OpenendedResponseWordCount"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    n_dv = merged.dropna(subset=["ic_qwenorpo400_logit", "DV_BeliefChange_Specific"]).shape[0]
    n_canon = merged.dropna(subset=["ic_qwenorpo400_logit", "DV_BeliefChange_Specific",
                                    "Pre_Belief_Specific", "OpenendedResponseWordCount"]).shape[0]

    print("SI Note 9 — Costello control-arm Ns (length-fixed IC)")
    print(f"  full active control arm (IC-scored set)        = {full:>5d}   [manuscript: 1,152]")
    print(f"  control rows merged with per-participant IC    = {n_merged:>5d}   [manuscript: 1,107]")
    print(f"  + post-dialogue belief re-measurement (DV)     = {n_dv:>5d}   [manuscript: 1,075]")
    print(f"  + canonical placebo spec (DV+pre+wc)           = {n_canon:>5d}")
    print()
    print("  AllData raw control rows (dedup participantId)  = {:>5d}".format(len(ctrl)))


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    (Path(__file__).parent / "note09_control_n_audit_output.txt").write_text(text)
