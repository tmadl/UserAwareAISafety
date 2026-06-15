#!/usr/bin/env python3
"""note_quintile_examples.py — build the IC-quintile example table for the SI.

For interim face validity (in lieu of not-yet-performed human IC coding of the
Costello-specific texts), select a representative pre-treatment statement from
each IC quintile of the headline analysis sample (N=1782), so readers can see
what low- vs high-IC text looks like under the primary Qwen3.5-ORPO-400 logit-EV
scorer.

Selection rule: within each quintile, the candidate whose IC is closest to the
quintile-median IC (ties → shorter text), among texts of moderate length
(20-120 words) so the excerpt is self-contained and not a one-liner. Prints
several candidates per quintile so a neutral, non-graphic exemplar can be chosen
by hand for the LaTeX table.

Output: prints candidates; writes note_quintile_examples_output.txt.
"""
import io, json, sys, re
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
REL = HERE.parent


def wc(t):
    return len(str(t).split())


def run():
    rows = [json.loads(l) for l in open(REL / "data/costello2024/texts_for_scoring.jsonl")]
    txt = pd.DataFrame({"participantId": [r["participantId"] for r in rows],
                        "text_initial": [r["text_initial"] for r in rows]})
    q = pd.read_csv(REL / "data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv")
    q = q[["participantId", "ic_qwenorpo400_logit"]].rename(columns={"ic_qwenorpo400_logit": "ic"})
    an = pd.read_csv(REL / "data/costello2024/analysis_data.csv")[
        ["participantId", "DV_BeliefChange_Specific", "Pre_Belief_Specific", "StudyNumber"]]

    d = an.merge(q, on="participantId").merge(txt, on="participantId")
    d = d.dropna(subset=["ic", "DV_BeliefChange_Specific", "Pre_Belief_Specific"]).copy()
    print(f"analysis sample N = {len(d)}")

    d["q"] = pd.qcut(d["ic"], 5, labels=[1, 2, 3, 4, 5])
    d["wc"] = d["text_initial"].map(wc)
    print("\nquintile IC ranges (logit-EV):")
    for qi in [1, 2, 3, 4, 5]:
        s = d[d["q"] == qi]["ic"]
        print(f"  Q{qi}: n={len(s)}  IC [{s.min():.2f}, {s.max():.2f}]  median {s.median():.2f}  "
              f"mean DV {d[d['q']==qi]['DV_BeliefChange_Specific'].mean():.1f}")

    for qi in [1, 2, 3, 4, 5]:
        sub = d[(d["q"] == qi) & (d["wc"].between(20, 120))].copy()
        med = sub["ic"].median()
        sub["dist"] = (sub["ic"] - med).abs()
        sub = sub.sort_values(["dist", "wc"]).head(4)
        print(f"\n{'='*78}\nQ{qi} candidates (near-median IC, 20-120 words)\n{'='*78}")
        for _, r in sub.iterrows():
            excerpt = re.sub(r"\s+", " ", str(r["text_initial"])).strip()
            print(f"\n  IC={r['ic']:.2f}  DV={r['DV_BeliefChange_Specific']:+.0f}  "
                  f"pre={r['Pre_Belief_Specific']:.0f}  wc={r['wc']}  study={int(r['StudyNumber'])}  "
                  f"pid={r['participantId'][:8]}")
            print(f"    {excerpt[:320]}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    t = buf.getvalue(); sys.stdout.write(t)
    (HERE / "note_quintile_examples_output.txt").write_text(t)
