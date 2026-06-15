#!/usr/bin/env python3
"""note25b_ih_v2_migration.py — authoritative numbers for the IH-scorer-v2
migration (replaces ckpt100 as the paper's validated text-IH instrument).

IH v2 source (pre-release, outside this repository):
  /mnt/workvm/UserAwareAISafety/_orpo/ih2/costello_texts_for_scoring_initial_ih_scores.csv
  (pre-treatment text, 3-seed ensemble EV `ens_ev_mean`; joined to
  participantId via exact text match against costello_combined_ih_scores.csv)

Prints every quantity the manuscript edits need:
  - frame construction with explicit duplicate-participantId accounting
  - r matrix: IC_q400 x IH_v2 x IH_ck100 x rubric-IH
  - standalone quadratic (note25 fit_standalone): v2 vs ckpt100 vs rubric
  - residual-on-IC quadratic (note25 fit_residual): v2 vs ckpt100 vs rubric
  - linear-only model: DV ~ z(IH) + pre + wc
  - attenuation factor rubric-residual-BF / v2-residual-BF
  - Study-1 Likert self-report convergence (Pearson + Spearman)

Output: prints; writes note25b_ih_v2_migration_output.txt next to this script.
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
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"
IH2 = Path("/mnt/workvm/UserAwareAISafety/_orpo/ih2")
LIKERT_CSV = Path("/mnt/workvm/UserAwareAISafety/data/costello2024/"
                  "costello_ih_ckpt100_actuallyused_in_paper.csv")

spec25 = importlib.util.spec_from_file_location(
    "n25", HERE / "note25_alt_constructs_discriminant.py")
n25 = importlib.util.module_from_spec(spec25)
spec25.loader.exec_module(n25)


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def build_frame():
    an = pd.read_csv(DATA / "analysis_data.csv")
    ic_old = pd.read_csv(DATA / "all_complexity_scores.csv").rename(
        columns={"IC_openai": "IC"})
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic_q400 = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC_q400": q["ic_qwenorpo400_logit"].astype(float).values})
    alt = pd.read_csv(DATA / "alt_constructs_logitev.csv")[
        ["participantId", "IH_ev"]].rename(columns={"IH_ev": "IH_rubric"})
    ih_ck = pd.read_csv(DATA / "ih_aot_prototypes/preds_ih_guo_only_decomp_ckpt100.csv")
    ih_ck = ih_ck[ih_ck["has_text"] == True].groupby(
        "participantId", as_index=False)["pred_ih_ev"].mean().rename(
        columns={"pred_ih_ev": "IH_ck100"})

    ini = pd.read_csv(IH2 / "costello_texts_for_scoring_initial_ih_scores.csv")
    comb = pd.read_csv(IH2 / "costello_combined_ih_scores.csv")
    ini["key"] = ini["Paragraph"].astype(str).str.strip()
    comb["key"] = comb["text_initial"].astype(str).str.strip()
    ihm = ini.merge(comb[["participantId", "key"]].drop_duplicates("key"),
                    on="key", how="inner")
    ih_v2 = ihm[["participantId", "ens_ev_mean"]].rename(
        columns={"ens_ev_mean": "IH_v2"}).drop_duplicates("participantId")
    print(f"IH v2 text-match: {len(ihm)} of {len(ini)} initial texts matched; "
          f"{len(ih_v2)} unique participantIds")

    df = (an.merge(ic_old[["participantId", "IC"]], on="participantId", how="inner")
            .merge(ic_q400, on="participantId", how="left")
            .merge(alt, on="participantId", how="left")
            .merge(ih_ck, on="participantId", how="left")
            .merge(ih_v2, on="participantId", how="left"))
    n_before = len(df)
    dup = df["participantId"].duplicated().sum()
    df = df.drop_duplicates("participantId").reset_index(drop=True)
    print(f"Frame: {n_before} rows; duplicate participantIds dropped: {dup}; "
          f"final N = {len(df)}")
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "IC", "IC_q400",
              "IH_rubric", "IH_ck100", "IH_v2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def lin_only(df, col):
    sub = df.dropna(subset=["DV_BeliefChange_Specific", col,
                            "Pre_Belief_Specific", "OpenendedResponseWordCount"])
    y = sub["DV_BeliefChange_Specific"].values
    X = sm.add_constant(np.column_stack([
        zs(sub[col]), zs(sub["Pre_Belief_Specific"]),
        zs(sub["OpenendedResponseWordCount"])]))
    m = sm.OLS(y, X).fit()
    return len(sub), m.params[1], m.pvalues[1]


def run():
    df = build_frame()
    print()
    print("=== descriptives / correlations ===")
    for c in ["IH_v2", "IH_ck100", "IH_rubric"]:
        print(f"{c}: n={df[c].notna().sum()}, mean={df[c].mean():.2f}, "
              f"SD={df[c].std():.2f}")
    for a, b in [("IC_q400", "IH_v2"), ("IC_q400", "IH_ck100"),
                 ("IC_q400", "IH_rubric"), ("IH_v2", "IH_ck100"),
                 ("IH_v2", "IH_rubric")]:
        cc = df.dropna(subset=[a, b])
        print(f"r({a}, {b}) = {cc[a].corr(cc[b]):.3f}  (n={len(cc)})")

    print()
    print("=== standalone quadratic (DV ~ z(X)+z(X^2)+pre+wc) ===")
    for label, col in [("IH v2", "IH_v2"), ("IH ckpt100", "IH_ck100"),
                       ("IH rubric (gpt-4.1-mini)", "IH_rubric")]:
        s = n25.fit_standalone(df, col)
        print(f"{label:26s}: n={s['n']}, b_lin={s['b_lin']:+.2f} "
              f"(p={s['p_lin']:.3f}), b_quad={s['b_quad']:+.2f} "
              f"(p={s['p_quad']:.3f}), BF_quad={s['bf']:.3g}")

    print()
    print("=== residual-on-IC_q400 quadratic ===")
    bfs = {}
    for label, col in [("IH v2", "IH_v2"), ("IH ckpt100", "IH_ck100"),
                       ("IH rubric (gpt-4.1-mini)", "IH_rubric")]:
        r = n25.fit_residual(df, col, "IC_q400")
        bfs[col] = r["bf"]
        print(f"{label:26s}: n={r['n']}, b_lin={r['b_lin']:+.2f} "
              f"(p={r['p_lin']:.3f}), b_quad={r['b_quad']:+.2f} "
              f"(p={r['p_quad']:.3f}), BF_quad={r['bf']:.3g}")
    print(f"attenuation rubric->v2: {bfs['IH_rubric'] / bfs['IH_v2']:,.0f}x "
          f"(rubric BF {bfs['IH_rubric']:.3g} / v2 BF {bfs['IH_v2']:.3g})")
    print("NOTE: rubric residualisation here uses IC_q400; the published "
          "BF=285 residualises rubric-IH on gpt-4.1-mini full-dialogue IC "
          "(same-channel pairing) — see note25 script for that pairing.")

    print()
    print("=== linear-only model (DV ~ z(IH)+pre+wc) ===")
    for label, col in [("IH v2", "IH_v2"), ("IH ckpt100", "IH_ck100")]:
        n, b, p = lin_only(df, col)
        print(f"{label:12s}: n={n}, b_lin={b:+.2f}, p={p:.5f}")

    print()
    print("=== Study-1 Likert self-report convergence ===")
    lik = pd.read_csv(LIKERT_CSV)
    lik = lik.dropna(subset=["IH"]).drop_duplicates("participantId")[
        ["participantId", "IH"]].rename(columns={"IH": "IH_likert"})
    cc = df.merge(lik, on="participantId", how="inner")
    for label, col in [("IH v2", "IH_v2"), ("IH ckpt100", "IH_ck100")]:
        sub = cc.dropna(subset=[col, "IH_likert"])
        r_p, p_p = pearsonr(sub[col], sub["IH_likert"])
        r_s, p_s = spearmanr(sub[col], sub["IH_likert"])
        print(f"{label:12s}: n={len(sub)}, Pearson r={r_p:.3f} (p={p_p:.4f}), "
              f"Spearman rho={r_s:.3f} (p={p_s:.4f})")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    (HERE / "note25b_ih_v2_migration_output.txt").write_text(text)
    sys.stdout.write(f"\nWrote {HERE / 'note25b_ih_v2_migration_output.txt'}\n")
