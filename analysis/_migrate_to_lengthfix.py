#!/usr/bin/env python3
"""_migrate_to_lengthfix.py — migrate the release's Qwen3-ORPO-400 IC scores to
the length-bug-fixed rescoring (2026-05-30).

The IC scorer silently truncated extremely long prompts (right-truncation dropped
the assistant generation marker, so logits were read at a garbage position). The
fix pre-clips user text and sets truncation_side="left". Drift is trivial
(Pearson r >= 0.9915 every dataset; Costello headline beta moved 0.8%), but we
adopt the fixed scores as canonical for compatibility.

Subtlety: the fixed CSVs carry participantId + row_idx but are re-sorted by
length-bin; sorting by row_idx matches the original jsonl order only ~99.95%
(one row drifts). The release's row-order joins therefore must NOT be fed the
fixed files directly. Instead, for each consumer we REBUILD the release file by
participantId lookup into the exact row order that consumer expects, so the
existing (row-order) loaders stay byte-for-byte correct AND use fixed scores.

Run once; idempotent. Prints a per-file report.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

REL = Path(__file__).resolve().parent.parent
FIX = Path("/mnt/workvm/UserAwareAISafety/data/IC_scored_length_bug_fixed")
DCOST = REL / "data" / "costello2024"
DQ400 = REL / "data" / "ic_qwen3orpo400"
DTESS = REL / "data" / "tessler2024"
DCHENG = REL / "data" / "cheng2006"
DSALVI = REL / "data" / "salvi2005"


def rebuild_in_jsonl_order(jsonl, fixed_csv, out_csv, logit_col="ic_qwenorpo400_logit",
                           greedy_col="ic_qwenorpo400_greedy"):
    """Write out_csv in the participantId order of `jsonl`, scores looked up by
    participantId from fixed_csv. Preserves legacy column names so row-order
    loaders that read q['ic_qwenorpo400_logit'] by position stay correct."""
    meta = [json.loads(l) for l in open(jsonl)]
    fx = pd.read_csv(fixed_csv).set_index("participantId")
    pids = [m["participantId"] for m in meta]
    rows = {"participantId": pids,
            "ic_qwenorpo400_logit": [fx[logit_col].get(p) for p in pids]}
    if greedy_col in fx.columns:
        rows["ic_qwenorpo400"] = [fx[greedy_col].get(p) for p in pids]
    out = pd.DataFrame(rows)
    n_missing = out["ic_qwenorpo400_logit"].isna().sum()
    out.to_csv(out_csv, index=False)
    return len(out), n_missing


def main():
    report = []

    # 1) Row-order consumers: REBUILD in texts_for_scoring.jsonl order by
    #    participantId lookup. These scripts (01/06/02_cheng/03_salvi/...) zip the
    #    fixed CSV positionally with the jsonl meta (and assert equal length), so
    #    the length-bin-sorted fixed file must NOT be copied directly. Rebuilding
    #    restores the jsonl row count; participants whose text was empty/unscored
    #    (absent from the fixed file) get NaN — every consumer dropna's the IC col.
    for label, jsonl, src, out in [
        ("costello primary", DCOST / "texts_for_scoring.jsonl",
         FIX / "costello_texts_for_scoring_initial_qwenorpo400.csv",
         DQ400 / "costello_texts_for_scoring_initial_qwenorpo400.csv"),
        ("cheng", DCHENG / "texts_for_scoring.jsonl",
         FIX / "cheng_texts_for_scoring_initial_qwenorpo400.csv",
         DQ400 / "cheng_texts_for_scoring_initial_qwenorpo400.csv"),
        ("salvi", DSALVI / "texts_for_scoring.jsonl",
         FIX / "salvi_texts_for_scoring_initial_qwenorpo400.csv",
         DQ400 / "salvi_texts_for_scoring_initial_qwenorpo400.csv"),
    ]:
        n, miss = rebuild_in_jsonl_order(jsonl, src, out)
        report.append((f"{label} (jsonl-order rebuild)", n, f"{miss} NaN"))

    # 2) participantId-keyed files: copy fixed directly (consumers join by ID)
    for name, dest in [
        ("costello_controls_qwenorpo400.csv", DCOST),
        ("costello_gpt_ic_qwenorpo400.csv", DCOST),
        ("texts_for_scoring_all_qwenorpo400_turns.csv", DCOST),
        ("tessler_ic_qwenorpo400.csv", DTESS),
        ("boissin_texts_for_scoring_qwenorpo400.jsonl", DQ400),
    ]:
        src = FIX / name
        if src.exists():
            shutil.copy2(src, dest / name)
            report.append((f"copied {name}", "ok", ""))
        else:
            report.append((f"MISSING SOURCE {name}", "--", ""))

    print("Length-fix migration report:")
    for r in report:
        print("  ", r)


if __name__ == "__main__":
    main()
