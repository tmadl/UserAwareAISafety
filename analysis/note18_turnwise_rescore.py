#!/usr/bin/env python3
"""note18_turnwise_rescore.py — regenerate the Costello per-turn + cumulative IC
scores with the LENGTH-BUG-FIXED Qwen3.5-ORPO-400 logit-EV scorer.

WHY THIS EXISTS
---------------
The turns file `data/costello2024/texts_for_scoring_all_qwenorpo400_turns.csv`
was (a) originally scored before the scorer's long-text clip fix, so the long
cumulative windows (cum_3 / cum_4 = multi-turn dialogue) were truncated to
garbage; and (b) further corrupted by the length-fix migration, which dropped
the per-turn texts and misaligned the per-turn score columns (turn_1/turn_3 IC
means came out ~5.2-5.6, impossible for short single turns). The original turn
TEXTS are intact in the canonical `texts_for_scoring.jsonl` (`text_all`), so this
script re-scores them cleanly with `score_texts` (which clips user text on the
left while preserving the generation prompt — the actual length-bug fix) and
rewrites the turns file.

TURN / WINDOW CONVENTION  (verified: `" ||| ".join(segments) == text_all`)
-------------------------------------------------------------------------
  Paragraph              = the user turns joined by " ||| "  (== text_all)
  turn_k_ic_logit        = IC of user turn k ALONE          (k = 0..3; NaN > n_turns)
  cum_k_ic_logit         = IC of the first k turns joined by " ||| "
                           (cum_1 = turn_0 alone;  cum_4 = full dialogue = text_all)

SUBSET
------
Reproduces the exact 1,242-participant turnwise subset of the published analysis:
the participants present in the original turns file (preserved in the
`*.prebugfix.bak`), matched to the canonical jsonl by the first user turn.
Texts come from the canonical jsonl `text_all`; the .bak is used ONLY to define
subset membership.

CONSUMERS (unchanged): `analysis/note18_turnwise_ic_stability.py` and
`scripts/review_qwen3orpo400_costello_turnwise.py` — both match turns rows to
`analysis_data` by first-segment text, so `Paragraph` is preserved verbatim.

RUN
---
On a GPU host with the IC-scorer-q400 adapter (see the adapter's
`inference_example.py`; 4-bit needs >=24 GB VRAM). Edit ADAPTER / paths below if
the server layout differs. ~1,242 participants -> ~8-9k unique texts -> ~10-20 min.

    python3 note18_turnwise_rescore.py

After it writes the new turns CSV, re-run `analysis/note18_turnwise_ic_stability.py`
to regenerate tab:turnwise_descriptives + tab:turnwise_fits, and reconcile the
SI Note 18 table + main-text turn-by-turn claim (BF<=0.03 for later windows).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths (edit for the server if needed) --------------------------------
ADAPTER = "/mnt/workvm/UserAwareAISafety/hf_upload/ic-scorer-q400"
REL = Path("/mnt/workvm/UserAwareAISafety_release")
JSONL = REL / "data/costello2024/texts_for_scoring.jsonl"
BAK = REL / "data/costello2024/texts_for_scoring_all_qwenorpo400_turns.csv.prebugfix.bak"
OUT = REL / "data/costello2024/texts_for_scoring_all_qwenorpo400_turns.csv"

SEP = " ||| "
SPLIT = re.compile(r"\s*\|\|\|\s*")

sys.path.insert(0, ADAPTER)
from inference_example import score_texts  # noqa: E402  (logit-EV, length-fixed)


def segments_of(text_all):
    return [s for s in SPLIT.split(str(text_all).strip()) if s]


def main():
    # canonical texts
    rows = [json.loads(l) for l in open(JSONL)]
    by_init = {}                       # first-turn text -> (participantId, text_all)
    for r in rows:
        by_init[str(r["text_initial"]).strip()] = (r["participantId"], r["text_all"])

    # subset membership from the original turns file (preserved in the .bak)
    bak = pd.read_csv(BAK)
    subset = []                        # list of (participantId, text_all)
    unmatched = 0
    for para in bak["Paragraph"]:
        first = SPLIT.split(str(para).strip())[0].strip()
        hit = by_init.get(first)
        if hit is None:
            unmatched += 1
            continue
        subset.append(hit)
    print(f".bak rows = {len(bak)}; matched to jsonl = {len(subset)}; unmatched = {unmatched}")

    # collect every distinct text to score once (turns + cumulative windows)
    seg_lists = [segments_of(ta) for _, ta in subset]
    to_score = set()
    for segs in seg_lists:
        for k in range(len(segs)):
            to_score.add(segs[k])              # turn_k
            to_score.add(SEP.join(segs[: k + 1]))  # cum_{k+1}
    texts = sorted(to_score)
    print(f"unique texts to score = {len(texts)}")

    ev = score_texts(texts)                    # length-fixed logit-EV in [1,7]
    score = dict(zip(texts, ev))

    out = []
    for (pid, ta), segs in zip(subset, seg_lists):
        n = len(segs)
        rec = {"participantId": pid, "Paragraph": ta, "n_turns": n}
        for k in range(4):
            rec[f"turn_{k}_ic_logit"] = score[segs[k]] if k < n else np.nan
        for k in range(1, 5):
            rec[f"cum_{k}_ic_logit"] = score[SEP.join(segs[:k])] if k <= n else np.nan
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")

    # sanity checks
    d = (df["turn_0_ic_logit"] - df["cum_1_ic_logit"]).abs()
    print(f"sanity  turn_0 == cum_1 : max|diff| = {d.max():.5f}  (should be ~0)")
    for k in range(4):
        v = df[f"turn_{k}_ic_logit"].dropna()
        print(f"  turn_{k}: n={len(v)}  mean IC={v.mean():.2f}  SD={v.std():.2f}")
    for k in range(1, 5):
        v = df[f"cum_{k}_ic_logit"].dropna()
        print(f"  cum_{k} : n={len(v)}  mean IC={v.mean():.2f}  SD={v.std():.2f}")
    print("\nNext: re-run analysis/note18_turnwise_ic_stability.py and reconcile "
          "the SI Note 18 table + main-text turn-by-turn claim.")


if __name__ == "__main__":
    main()
