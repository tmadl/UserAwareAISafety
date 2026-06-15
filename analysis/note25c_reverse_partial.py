#!/usr/bin/env python3
"""note25c_reverse_partial.py — reverse-partial discriminant test (reviewer M5).

Note 25 partials IC OUT OF the alternative constructs and shows their residual
curvature collapses. This script runs the REVERSE: residualise IC (Q400
logit-EV) on each alternative, and test whether IC's residual quadratic
survives. If it does, the curvature is IC-specific; if it collapses
symmetrically, the data identify a shared text-scored sophistication
dimension rather than IC per se.

Partials: rubric AOT/NFC/OMI (gpt-4.1-mini channel), validated IH-scorer-v2.
Also prints the full correlation matrix (to verify the claimed r(IC,NFC))
and the forward partials for side-by-side reference.

Output: prints; writes note25c_reverse_partial_output.txt next to this script.
"""
import importlib.util
import io
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "costello2024"
DATA_Q = HERE.parent / "data" / "ic_qwen3orpo400"
IH2 = Path("/mnt/workvm/UserAwareAISafety/_orpo/ih2")

spec25 = importlib.util.spec_from_file_location(
    "n25", HERE / "note25_alt_constructs_discriminant.py")
n25 = importlib.util.module_from_spec(spec25)
spec25.loader.exec_module(n25)


def build_frame():
    an = pd.read_csv(DATA / "analysis_data.csv")
    ic_old = pd.read_csv(DATA / "all_complexity_scores.csv").rename(
        columns={"IC_openai": "IC_gptfull"})
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic_q400 = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC_q400": q["ic_qwenorpo400_logit"].astype(float).values})
    alt = pd.read_csv(DATA / "alt_constructs_logitev.csv")[
        ["participantId", "AOT_ev", "NFC_ev", "OMI_ev", "IH_ev"]].rename(
        columns={"AOT_ev": "AOT_rub", "NFC_ev": "NFC_rub", "OMI_ev": "OMI_rub",
                 "IH_ev": "IH_rubric"})
    ini = pd.read_csv(IH2 / "costello_texts_for_scoring_initial_ih_scores.csv")
    comb = pd.read_csv(IH2 / "costello_combined_ih_scores.csv")
    ini["key"] = ini["Paragraph"].astype(str).str.strip()
    comb["key"] = comb["text_initial"].astype(str).str.strip()
    ihm = ini.merge(comb[["participantId", "key"]].drop_duplicates("key"),
                    on="key", how="inner")
    ih_v2 = ihm[["participantId", "ens_ev_mean"]].rename(
        columns={"ens_ev_mean": "IH_v2"}).drop_duplicates("participantId")

    df = (an.merge(ic_old[["participantId", "IC_gptfull"]],
                   on="participantId", how="inner")
            .merge(ic_q400, on="participantId", how="left")
            .merge(alt, on="participantId", how="left")
            .merge(ih_v2, on="participantId", how="left"))
    df = df.drop_duplicates("participantId").reset_index(drop=True)
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "IC_q400", "IC_gptfull",
              "AOT_rub", "NFC_rub", "OMI_rub", "IH_rubric", "IH_v2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def run():
    df = build_frame()
    print(f"N = {len(df)}")
    print()
    cols = ["IC_q400", "AOT_rub", "NFC_rub", "OMI_rub", "IH_rubric", "IH_v2"]
    print("=== correlation matrix ===")
    print(df[cols].corr().round(3).to_string())
    print()

    print("=== IC reference (standalone quadratic) ===")
    s = n25.fit_standalone(df, "IC_q400")
    print(f"IC_q400: n={s['n']}, b_quad={s['b_quad']:+.2f} "
          f"(p={s['p_quad']:.4f}), BF={s['bf']:.3g}")
    print()

    print("=== REVERSE partials: IC_q400 residualised on each alternative ===")
    for alt in ["NFC_rub", "AOT_rub", "OMI_rub", "IH_rubric", "IH_v2"]:
        r = n25.fit_residual(df, "IC_q400", alt)
        print(f"IC | {alt:9s}: n={r['n']}, b_lin={r['b_lin']:+.2f} "
              f"(p={r['p_lin']:.3f}), b_quad={r['b_quad']:+.2f} "
              f"(p={r['p_quad']:.4f}), BF_quad={r['bf']:.3g}")
    print()

    print("=== FORWARD partials (reference; published direction) ===")
    for alt in ["NFC_rub", "AOT_rub", "OMI_rub", "IH_rubric", "IH_v2"]:
        r = n25.fit_residual(df, alt, "IC_q400")
        print(f"{alt:9s} | IC: n={r['n']}, b_quad={r['b_quad']:+.2f} "
              f"(p={r['p_quad']:.4f}), BF_quad={r['bf']:.3g}")

    print()
    print("=== MATCHED-WINDOW/CHANNEL: gpt-full-dialogue IC as the IC measure ===")
    print("(rubric alternatives are scored on full dialogue via gpt-4.1-mini,")
    print(" so IC_gptfull | alt and alt | IC_gptfull are the symmetric pair)")
    print("-- positive control: FORWARD partials, should reproduce published")
    print("   values (NFC|IC: BF 0.03; AOT 0.59; OMI 1.5; IH_rubric 285) --")
    for alt in ["NFC_rub", "AOT_rub", "OMI_rub", "IH_rubric"]:
        r = n25.fit_residual(df, alt, "IC_gptfull")
        print(f"{alt:9s} | IC_gptfull: n={r['n']}, b_quad={r['b_quad']:+.2f} "
              f"(p={r['p_quad']:.4f}), BF_quad={r['bf']:.3g}")
    print("-- REVERSE: IC_gptfull residualised on each alternative --")
    for alt in ["NFC_rub", "AOT_rub", "OMI_rub", "IH_rubric"]:
        r = n25.fit_residual(df, "IC_gptfull", alt)
        print(f"IC_gptfull | {alt:9s}: n={r['n']}, b_quad={r['b_quad']:+.2f} "
              f"(p={r['p_quad']:.4f}), BF_quad={r['bf']:.3g}")
    print(f"r(IC_gptfull, NFC_rub) = "
          f"{df['IC_gptfull'].corr(df['NFC_rub']):.3f}; "
          f"r(IC_gptfull, IC_q400) = {df['IC_gptfull'].corr(df['IC_q400']):.3f}")

    print()
    print("=== joint model: IC + IC^2 + NFC + NFC^2 (head-to-head) ===")
    import statsmodels.api as sm
    sub = df.dropna(subset=["DV_BeliefChange_Specific", "IC_q400", "NFC_rub",
                            "Pre_Belief_Specific",
                            "OpenendedResponseWordCount"]).copy()

    def zs(x):
        x = np.asarray(x, float)
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

    y = sub["DV_BeliefChange_Specific"].values
    ic = zs(sub["IC_q400"]); nfc = zs(sub["NFC_rub"])
    pre = zs(sub["Pre_Belief_Specific"]); wc = zs(sub["OpenendedResponseWordCount"])
    X = sm.add_constant(np.column_stack([ic, ic**2, nfc, nfc**2, pre, wc]))
    m = sm.OLS(y, X).fit()
    print(f"n={len(sub)}: IC^2 b={m.params[2]:+.2f} (p={m.pvalues[2]:.4f}); "
          f"NFC^2 b={m.params[4]:+.2f} (p={m.pvalues[4]:.4f})")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    (HERE / "note25c_reverse_partial_output.txt").write_text(text)
    sys.stdout.write(f"\nWrote {HERE / 'note25c_reverse_partial_output.txt'}\n")
