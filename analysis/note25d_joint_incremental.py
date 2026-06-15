#!/usr/bin/env python3
"""note25d_joint_incremental.py — SI Note 25, joint incremental discriminant.

Autoreviewer detail #3: the residual-score diagnostic in Note 25
(residualise alt on IC, then test the squared residual) changes the
estimand, because the squared residual is not independent of nonlinear
functions of IC or of the controls. The reviewer asks for the direct
incremental model that includes IC, IC^2, the alternative construct, AND
its quadratic term simultaneously:

    DV ~ z(IC) + z(IC)^2 + z(Alt) + z(Alt)^2 + z(pre) + z(wc)

For each nearest-neighbour construct this reports:
  (1) does the IC^2 curvature SURVIVE the joint inclusion of Alt + Alt^2?
  (2) does Alt^2 add incremental value BEYOND IC + IC^2 (+ Alt)?
      - 1-df incremental BF for adding Alt^2 over {IC+IC^2+Alt+covs}
      - 1-df incremental F/p for the same comparison
  (3) 2-df incremental BF/F for adding {Alt+Alt^2} over {IC+IC^2+covs}

Channel pairing matches Note 25 / Table tab:alt_constructs:
  AOT, NFC, OMI, rubric-IH : full-dialogue gpt-4.1-mini IC reference (text_all)
  validated IH-v2          : Q400 pre-treatment IC reference (headline channel)

Headline z(IC)^2 scale throughout (predictor z-scored, then squared).
Output: prints; writes note25d_joint_incremental_output.txt next to this file.
"""
import io
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"
IH2 = Path("/mnt/workvm/UserAwareAISafety/_orpo/ih2")


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def build_frame():
    """Costello frame with both IC channels + all alternative scores + IH-v2.

    Mirrors note25/note25b construction exactly:
      IC        = IC_openai (gpt-4.1-mini, full-dialogue text_all)
      IC_q400   = Qwen3.5-ORPO-400 logit-EV (pre-treatment text_initial)
      AOT/IH_rubric/NFC/OMI = gpt-4.1-mini logit-EV rubric (full-dialogue)
      IH_v2     = construct-validated IH scorer, 3-seed ensemble (pre-treatment)
    """
    an = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)
    ic_old = pd.read_csv(DATA / "all_complexity_scores.csv").rename(
        columns={"IC_openai": "IC"})[["participantId", "IC"]]
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic_q400 = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC_q400": q["ic_qwenorpo400_logit"].astype(float).values})
    # NB: analysis_data.csv carries its own Likert "AOT"/"IH" columns, so the
    # text-derived rubric scores are renamed to *_txt to avoid a merge collision.
    alt = pd.read_csv(DATA / "alt_constructs_logitev.csv")[
        ["participantId", "AOT_ev", "IH_ev", "NFC_ev", "OMI_ev"]].rename(
        columns={"AOT_ev": "AOT_txt", "IH_ev": "IH_rubric",
                 "NFC_ev": "NFC_txt", "OMI_ev": "OMI_txt"})

    # validated IH-v2 (text-match against pre-treatment text, as in note25b)
    ini = pd.read_csv(IH2 / "costello_texts_for_scoring_initial_ih_scores.csv")
    comb = pd.read_csv(IH2 / "costello_combined_ih_scores.csv")
    ini["key"] = ini["Paragraph"].astype(str).str.strip()
    comb["key"] = comb["text_initial"].astype(str).str.strip()
    ihm = ini.merge(comb[["participantId", "key"]].drop_duplicates("key"),
                    on="key", how="inner")
    ih_v2 = ihm[["participantId", "ens_ev_mean"]].rename(
        columns={"ens_ev_mean": "IH_v2"}).drop_duplicates("participantId")

    df = (an.merge(ic_old, on="participantId", how="inner")
            .merge(ic_q400, on="participantId", how="left")
            .merge(alt, on="participantId", how="left")
            .merge(ih_v2, on="participantId", how="left"))
    df = df.drop_duplicates("participantId").reset_index(drop=True)
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "IC", "IC_q400",
              "AOT_txt", "IH_rubric", "NFC_txt", "OMI_txt", "IH_v2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def joint_model(df, ic_col, alt_col):
    """DV ~ z(IC)+z(IC)^2 + z(Alt)+z(Alt)^2 + z(pre)+z(wc), plus the two
    nested comparisons the reviewer asks for."""
    sub = df.dropna(subset=["DV_BeliefChange_Specific", ic_col, alt_col,
                            "Pre_Belief_Specific",
                            "OpenendedResponseWordCount"]).copy()
    y = sub["DV_BeliefChange_Specific"].values.astype(float)
    ic = zs(sub[ic_col].values); ic2 = ic ** 2
    al = zs(sub[alt_col].values); al2 = al ** 2
    pre = zs(sub["Pre_Belief_Specific"].values)
    wc = zs(sub["OpenendedResponseWordCount"].values)

    # nested models
    X_ic   = sm.add_constant(np.column_stack([ic, ic2, pre, wc]))          # IC+IC^2
    X_full = sm.add_constant(np.column_stack([ic, ic2, al, al2, pre, wc]))  # joint
    X_noQ  = sm.add_constant(np.column_stack([ic, ic2, al, pre, wc]))       # joint - Alt^2
    m_ic, m_full, m_noQ = (sm.OLS(y, X).fit() for X in (X_ic, X_full, X_noQ))

    # IC^2 in the joint model (does it survive?)
    b_ic2_joint, p_ic2_joint = m_full.params[2], m_full.pvalues[2]
    b_ic2_solo = m_ic.params[2]
    # Alt^2 in the joint model + 1-df incremental over {IC+IC^2+Alt}
    b_alt2, p_alt2 = m_full.params[4], m_full.pvalues[4]
    bf_alt2 = float(np.exp((m_noQ.bic - m_full.bic) / 2))   # >1 favours Alt^2
    F_alt2 = m_full.compare_f_test(m_noQ)                   # (F, p, df)
    # 2-df incremental {Alt+Alt^2} over {IC+IC^2}
    bf_altblock = float(np.exp((m_ic.bic - m_full.bic) / 2))
    F_altblock = m_full.compare_f_test(m_ic)
    return dict(
        n=len(sub),
        b_ic2_solo=b_ic2_solo, b_ic2_joint=b_ic2_joint, p_ic2_joint=p_ic2_joint,
        b_alt2=b_alt2, p_alt2=p_alt2, bf_alt2=bf_alt2,
        F_alt2=F_alt2[0], p_F_alt2=F_alt2[1],
        bf_altblock=bf_altblock, F_altblock=F_altblock[0], p_F_altblock=F_altblock[1],
        r_ic_alt=float(np.corrcoef(ic, al)[0, 1]))


def run():
    df = build_frame()
    print("SI Note 25 — joint incremental discriminant model")
    print("DV ~ z(IC) + z(IC)^2 + z(Alt) + z(Alt)^2 + z(pre) + z(wc)\n")

    rows = [
        ("AOT  (rubric)        | gpt full-dialogue IC", "IC",      "AOT_txt"),
        ("IH   (rubric)        | gpt full-dialogue IC", "IC",      "IH_rubric"),
        ("NFC  (rubric)        | gpt full-dialogue IC", "IC",      "NFC_txt"),
        ("OMI  (rubric)        | gpt full-dialogue IC", "IC",      "OMI_txt"),
        ("IH-v2 (Guo-validated)| Q400 pre-treatment IC", "IC_q400", "IH_v2"),
    ]
    print(f"{'Construct | IC channel':<46}{'n':>5} {'b_IC2(joint)':>12} "
          f"{'p_IC2':>7} {'b_Alt2':>8} {'p_Alt2':>7} {'BF(Alt2|IC+IC2+Alt)':>20}")
    print("-" * 106)
    for label, ic_col, alt_col in rows:
        r = joint_model(df, ic_col, alt_col)
        print(f"{label:<46}{r['n']:>5} {r['b_ic2_joint']:>+12.3f} "
              f"{r['p_ic2_joint']:>7.3f} {r['b_alt2']:>+8.3f} {r['p_alt2']:>7.3f} "
              f"{r['bf_alt2']:>20.3g}")
    print()
    print("Detail (incremental tests):")
    for label, ic_col, alt_col in rows:
        r = joint_model(df, ic_col, alt_col)
        print(f"  {label}")
        print(f"     IC^2: solo b={r['b_ic2_solo']:+.3f} -> joint b={r['b_ic2_joint']:+.3f} "
              f"(p={r['p_ic2_joint']:.3f});  r(IC,Alt)={r['r_ic_alt']:+.2f}")
        print(f"     Alt^2 incremental over IC+IC^2+Alt: F(1,df)={r['F_alt2']:.2f} "
              f"p={r['p_F_alt2']:.3f}  BF10={r['bf_alt2']:.3g}")
        print(f"     {{Alt+Alt^2}} block over IC+IC^2: F(2,df)={r['F_altblock']:.2f} "
              f"p={r['p_F_altblock']:.3f}  BF10={r['bf_altblock']:.3g}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    t = buf.getvalue()
    sys.stdout.write(t)
    (HERE / "note25d_joint_incremental_output.txt").write_text(t)
    sys.stdout.write(f"\nWrote {HERE / 'note25d_joint_incremental_output.txt'}\n")
