#!/usr/bin/env python3
"""note17_ic_distribution.py — SI Note 17.

Compares the Costello IC distribution to two reference corpora (Jakob
naturalistic online discourse and Suedfeld expert exemplars), all scored on
the same Q400 logit-EV pipeline. Reproduces SI Table tab:ic_dist plus the
distributional-difference statistics in Note 17 prose (Welch t, Levene F,
KS D, % above Jakob Q90, % below Jakob Q10).
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_Q = DATA / "ic_qwen3orpo400"


def descriptives(x, name, n_label=None):
    qs = np.percentile(x, [10, 25, 50, 75, 90])
    return dict(name=name, n=n_label or len(x), mean=np.mean(x), sd=np.std(x, ddof=1),
                q10=qs[0], q25=qs[1], median=qs[2], q75=qs[3], q90=qs[4])


def main():
    # === Costello (pre-treatment, Q400 logit-EV) ===
    meta = [json.loads(l) for l in open(DATA / "costello2024" / "texts_for_scoring.jsonl")]
    q_cos = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    an = pd.read_csv(DATA / "costello2024" / "analysis_data.csv")
    pid_cos = [m["participantId"] for m in meta]
    cos_df = pd.DataFrame({"participantId": pid_cos,
                           "IC": q_cos["ic_qwenorpo400_logit"].astype(float).values})
    # Restrict to Costello primary analysis sample
    cos = cos_df.merge(an[["participantId"]], on="participantId", how="inner")
    cos_ic = cos["IC"].dropna().values

    # === Jakob and Suedfeld (Q400 logit-EV from validation_results) ===
    val = pd.read_csv(DATA / "ic_validation" / "validation_results_qwen3orpo400.csv")
    jak_ic = val[val["source"] == "jakob"]["logit_score"].astype(float).values
    sf_ic = val[val["source"] == "psuedfeld"]["logit_score"].astype(float).values

    # === Print descriptives table (tab:ic_dist) ===
    print("SI Note 17 — IC distribution comparison (Q400 logit-EV)\n")
    print(f"{'Sample':<35s} {'N':>6s}  {'Mean':>5s}  {'SD':>5s}  {'Q10':>5s}  {'Q25':>5s}  {'Median':>6s}  {'Q75':>5s}  {'Q90':>5s}")
    print("-" * 100)
    for label, x in [("Costello (pre-treatment)", cos_ic),
                     ("Jakob online-discourse",   jak_ic),
                     ("Suedfeld expert exemplars", sf_ic if sf_ic is not None else np.array([]))]:
        if len(x) == 0:
            continue
        d = descriptives(x, label)
        print(f"{label:<35s} {d['n']:>6d}  {d['mean']:>5.2f}  {d['sd']:>5.2f}  "
              f"{d['q10']:>5.2f}  {d['q25']:>5.2f}  {d['median']:>6.2f}  {d['q75']:>5.2f}  {d['q90']:>5.2f}")

    # === Distributional comparison (Costello vs Jakob) ===
    print("\nDistributional comparison (Costello vs Jakob):")
    t, p_t = sp.ttest_ind(cos_ic, jak_ic, equal_var=False)
    F_lev, p_lev = sp.levene(cos_ic, jak_ic)
    D_ks, p_ks = sp.ks_2samp(cos_ic, jak_ic)
    print(f"  Welch t = {t:.1f}, p = {p_t:.3g}")
    print(f"  Levene F = {F_lev:.1f}, p = {p_lev:.3g}")
    print(f"  KS D = {D_ks:.3f}, p = {p_ks:.3g}")

    # % above Jakob Q90, % below Jakob Q10
    j_q10, j_q90 = np.percentile(jak_ic, [10, 90])
    pct_above = 100 * np.mean(cos_ic > j_q90)
    pct_below = 100 * np.mean(cos_ic < j_q10)
    print(f"  % of Costello above Jakob Q90 ({j_q90:.2f}): {pct_above:.1f}%")
    print(f"  % of Costello below Jakob Q10 ({j_q10:.2f}): {pct_below:.1f}%")


if __name__ == "__main__":
    main()
