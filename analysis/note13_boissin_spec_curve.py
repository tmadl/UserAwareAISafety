#!/usr/bin/env python3
"""note13_boissin_spec_curve.py — SI Note 13.

Specification-curve analysis for the Boissin linear-IC effect: 324
specifications varying DV (signed Δ vs |Δ|), IC source (text_all vs
text_initial), covariates (none vs +PreBelief vs +PreBelief+wc), subset
(9 levels: pooled, AI, Expert, Human-like, Neutral, Conspiracy, Epistemic,
AI×HL cell, movers), and outlier rule (none, 3 SD, 2 SD).

Reproduces the Note 13 aggregate stats:
  - Significant (p < .05): 22.5%
  - Direction-consistent: 91.4%
  - Sig + right direction: 22.5%
  - text_all source: 39% sig+right
  - text_initial source: 6% sig+right
  - Headline specifications table.
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import product
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "boissin2025"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"


def load():
    an = pd.read_csv(DATA / "analysis_data.csv")
    rows = [json.loads(l) for l in open(DATA_Q / "boissin_texts_for_scoring_qwenorpo400.jsonl")]
    ic = pd.DataFrame({
        "participantId": [r["participantId"] for r in rows],
        "IC_text_all":      [r["ic_qwenorpo400_all_logit"] for r in rows],
        "IC_text_initial":  [r["ic_qwenorpo400_initial_logit"] for r in rows],
        "wc_all":           [len(str(r["text_all"]).split()) for r in rows],
    })
    df = an.merge(ic, on="participantId", how="left")
    for c in ["belief_change", "PreBelief", "IC_text_all", "IC_text_initial", "wc_all"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fit_one(sub, dv_col, ic_col, covs):
    """Fit y ~ z(IC) + z(cov)... returns β_lin, p."""
    sub = sub.dropna(subset=[dv_col, ic_col] + list(covs)).copy()
    if len(sub) < 30:
        return None
    y = sub[dv_col].values.astype(float)
    ic_z = zs(sub[ic_col].values)
    cov_z = [zs(sub[c].values) for c in covs]
    X = sm.add_constant(np.column_stack([ic_z] + cov_z))
    try:
        m = sm.OLS(y, X).fit()
        return dict(n=len(sub), beta=m.params[1], p=m.pvalues[1])
    except Exception:
        return None


def main():
    df = load()
    df["abs_delta"] = df["belief_change"].abs()
    df["movers_mask"] = df["belief_change"].abs() > 0

    # Outlier rules: none, 3 SD, 2 SD on belief_change
    def apply_outlier(d, rule):
        if rule == "none":
            return d
        sd = d["belief_change"].std()
        mean = d["belief_change"].mean()
        cap = float(rule.split()[0]) * sd
        return d[(d["belief_change"] - mean).abs() <= cap]

    # Subsets
    subsets = {
        "pooled":        lambda d: d,
        "AI only":       lambda d: d[d["Speaker"] == "AI"],
        "Expert only":   lambda d: d[d["Speaker"] == "Expert"],
        "Human-like":    lambda d: d[d["PromptType"] == "Human-like"],
        "Neutral":       lambda d: d[d["PromptType"] == "Neutral"],
        "Conspiracy":    lambda d: d[d["BeliefType"] == "Conspiracy theories"],
        "Epistemic":     lambda d: d[d["BeliefType"] == "Epistemically unwarranted beliefs"],
        "AI × HL":       lambda d: d[(d["Speaker"] == "AI") & (d["PromptType"] == "Human-like")],
        "movers":        lambda d: d[d["movers_mask"]],
    }

    dvs = [("belief_change", "signed"),
           ("abs_delta",     "|Δ|")]
    ic_sources = [("IC_text_all",     "text_all"),
                  ("IC_text_initial", "text_initial")]
    cov_sets = [(),
                ("PreBelief",),
                ("PreBelief", "wc_all")]
    outlier_rules = ["none", "3 SD", "2 SD"]

    # Run all 2 × 2 × 3 × 9 × 3 = 324 specs
    results = []
    for (dv, dv_label), (ic, ic_label), covs, subset_name, ol in product(
            dvs, ic_sources, cov_sets, subsets.keys(), outlier_rules):
        sub_df = apply_outlier(subsets[subset_name](df), ol)
        if len(sub_df) < 30:
            continue
        r = fit_one(sub_df, dv, ic, covs)
        if r is None:
            continue
        # "Right direction": predicted resistance = β_lin > 0 on signed; β_lin < 0 on |Δ|
        right_dir = (r["beta"] > 0) if dv_label == "signed" else (r["beta"] < 0)
        results.append({
            "dv": dv_label, "ic_source": ic_label, "covs": "+".join(covs) or "none",
            "subset": subset_name, "outlier": ol,
            "n": r["n"], "beta": r["beta"], "p": r["p"],
            "sig": r["p"] < .05, "right_dir": right_dir,
            "sig_and_right": r["p"] < .05 and right_dir,
        })

    rdf = pd.DataFrame(results)
    print(f"SI Note 13 — Boissin specification curve\n")
    print(f"Total specifications run: {len(rdf)}")
    print(f"  Significant (p < .05):                       {rdf['sig'].sum():3d} / {len(rdf)} = "
          f"{100*rdf['sig'].mean():.1f}%")
    print(f"  Direction-consistent (resistance signature): {rdf['right_dir'].sum():3d} / {len(rdf)} = "
          f"{100*rdf['right_dir'].mean():.1f}%")
    print(f"  Significant AND right direction:             {rdf['sig_and_right'].sum():3d} / {len(rdf)} = "
          f"{100*rdf['sig_and_right'].mean():.1f}%")

    print("\nBy DV:")
    for dv in ["signed", "|Δ|"]:
        s = rdf[rdf["dv"] == dv]
        med_b = np.median(s["beta"])
        print(f"  {dv:<8s}: {100*s['sig_and_right'].mean():>5.0f}% sig+right, "
              f"median β = {med_b:+.2f}")

    print("\nBy IC source:")
    for src in ["text_all", "text_initial"]:
        s = rdf[rdf["ic_source"] == src]
        print(f"  {src:<14s}: {100*s['sig_and_right'].mean():>5.0f}% sig+right "
              f"({s['sig_and_right'].sum()} / {len(s)})")

    # Headline specifications table (subset of named-spec rows)
    print("\nHeadline specifications (text_all, no outlier filter, +PreBelief covs):")
    print(f"  {'DV':<8s} {'Subset':<28s} {'n':>5s}  {'β_lin':>8s}  {'p':>7s}")
    headline = rdf[(rdf["ic_source"] == "text_all") &
                   (rdf["outlier"] == "none") &
                   (rdf["covs"] == "PreBelief")]
    for subset_name in ["pooled", "AI only", "AI × HL", "movers"]:
        for dv in ["signed", "|Δ|"]:
            sub = headline[(headline["subset"] == subset_name) & (headline["dv"] == dv)]
            if len(sub) > 0:
                row = sub.iloc[0]
                print(f"  {dv:<8s} {subset_name:<28s} {row['n']:>5d}  "
                      f"{row['beta']:>+8.2f}  {row['p']:>7.3f}")


if __name__ == "__main__":
    main()
