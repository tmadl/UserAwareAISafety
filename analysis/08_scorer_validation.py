#!/usr/bin/env python3
"""
08_scorer_validation.py — IC scorer validation against human-coded corpora.

Reports agreement between the two scorers used in the paper (primary
Qwen3-ORPO-400 logit-EV; cross-scorer gpt-4.1-mini) and human expert IC codes
on two independent validation corpora:
  1. Suedfeld scoring manual exemplars (N=155, IC 1-7)
  2. Jakob et al. (2021) online discourse posts (N≈2281, IC 1-6)

Metrics: Pearson r, Spearman ρ, ICC(3,1), weighted κ, MAE, per-level accuracy.

Reads:
    data/ic_validation/suedfeld_scored.csv   (gpt-4.1-mini IC_openai)
    data/ic_validation/jakob_scored.csv      (gpt-4.1-mini IC_openai)
    data/ic_validation/validation_results_qwen3orpo400.csv  (Q400 logit-EV,
        columns: fold, source ∈ {jakob,suedfeld}, gt, logit_score, text)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ic_validation"


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def icc_3_1(human, model):
    """Two-way mixed, single-measures ICC(3,1) — consistency."""
    n = len(human)
    k = 2  # two raters
    data = np.column_stack([human, model])
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    grand_mean = data.mean()

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
    return icc


def weighted_kappa(human, model, weights="quadratic"):
    """Quadratic-weighted Cohen's kappa for ordinal data."""
    # Round model scores to nearest integer
    model_rounded = np.clip(np.round(model), 1, 7).astype(int)
    human_int = np.clip(np.round(human), 1, 7).astype(int)

    min_val = min(human_int.min(), model_rounded.min())
    max_val = max(human_int.max(), model_rounded.max())
    cats = list(range(min_val, max_val + 1))
    n_cats = len(cats)

    # Confusion matrix
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((n_cats, n_cats))
    for h, m in zip(human_int, model_rounded):
        O[cat_to_idx[h], cat_to_idx[m]] += 1
    O = O / O.sum()

    # Expected (outer product of marginals)
    E = np.outer(O.sum(axis=1), O.sum(axis=0))

    # Weight matrix
    W = np.zeros((n_cats, n_cats))
    for i in range(n_cats):
        for j in range(n_cats):
            if weights == "quadratic":
                W[i, j] = (cats[i] - cats[j]) ** 2 / (cats[-1] - cats[0]) ** 2
            else:
                W[i, j] = abs(cats[i] - cats[j]) / (cats[-1] - cats[0])

    kappa = 1 - (W * O).sum() / (W * E).sum()
    return kappa


def validate_corpus(df, human_col, name, ic_col="IC_openai"):
    """Run full validation suite on a scored corpus."""
    print(f"\n{'='*70}")
    print(f"  VALIDATION: {name}  (scorer: {ic_col})")
    print(f"{'='*70}")

    valid = df.dropna(subset=[ic_col, human_col]).copy()
    human = valid[human_col].values.astype(float)
    model = valid[ic_col].values

    print(f"\n  N = {len(valid)}")
    print(f"  Human IC: mean = {human.mean():.2f}, SD = {human.std():.2f}, "
          f"range = [{human.min():.1f}, {human.max():.1f}]")
    print(f"  LLM IC:   mean = {model.mean():.2f}, SD = {model.std():.2f}, "
          f"range = [{model.min():.1f}, {model.max():.1f}]")

    # Correlation
    r, p_r = sp.pearsonr(human, model)
    rho, p_rho = sp.spearmanr(human, model)
    print(f"\n  Pearson r    = {r:.3f}, p = {fmt_p(p_r)}")
    print(f"  Spearman ρ   = {rho:.3f}, p = {fmt_p(p_rho)}")

    # ICC
    icc = icc_3_1(human, model)
    print(f"  ICC(3,1)     = {icc:.3f}")

    # Weighted kappa
    kw = weighted_kappa(human, model, "quadratic")
    kl = weighted_kappa(human, model, "linear")
    print(f"  Weighted κ (quadratic) = {kw:.3f}")
    print(f"  Weighted κ (linear)    = {kl:.3f}")

    # MAE
    mae = np.mean(np.abs(human - model))
    rmse = np.sqrt(np.mean((human - model) ** 2))
    print(f"  MAE  = {mae:.3f}")
    print(f"  RMSE = {rmse:.3f}")

    # Per-level analysis
    print(f"\n  Per-level agreement:")
    print(f"  {'Human IC':>10s} {'N':>5s} {'LLM mean':>10s} {'LLM SD':>8s} {'MAE':>6s}")
    print(f"  {'-'*42}")
    for level in sorted(valid[human_col].unique()):
        sub = valid[valid[human_col] == level]
        m = sub[ic_col].mean()
        s = sub[ic_col].std()
        level_mae = np.mean(np.abs(sub[human_col].values - sub[ic_col].values))
        print(f"  {level:>10.1f} {len(sub):>5d} {m:>10.2f} {s:>8.2f} {level_mae:>6.2f}")

    # Exact and ±1 agreement (rounding model to nearest int)
    model_rounded = np.clip(np.round(model), 1, 7).astype(int)
    human_rounded = np.round(human).astype(int)
    exact = np.mean(model_rounded == human_rounded)
    within1 = np.mean(np.abs(model_rounded - human_rounded) <= 1)
    print(f"\n  Exact agreement (rounded): {exact:.1%}")
    print(f"  Within ±1 agreement:       {within1:.1%}")

    return {
        "n": len(valid),
        "r": r, "rho": rho,
        "icc": icc,
        "kw": kw,
        "mae": mae, "rmse": rmse,
        "exact": exact, "within1": within1,
    }


def main():
    print("Loading scored validation corpora...")

    results = {}

    # ── gpt-4.1-mini (cross-scorer baseline) ────────────────────────────
    sue_path = DATA / "suedfeld_scored.csv"
    if sue_path.exists():
        sue = pd.read_csv(sue_path)
        results["Suedfeld_gpt"] = validate_corpus(
            sue, "ic", "Suedfeld Exemplars", ic_col="IC_openai")

    jakob_path = DATA / "jakob_scored.csv"
    if jakob_path.exists():
        jakob = pd.read_csv(jakob_path)
        results["Jakob_gpt"] = validate_corpus(
            jakob, "ic_ordinal", "Jakob et al. (2021) Online Discourse",
            ic_col="IC_openai")

    # ── Primary: Qwen3-ORPO-400 logit-EV ────────────────────────────────
    q400_path = DATA / "validation_results_qwen3orpo400.csv"
    if q400_path.exists():
        q400 = pd.read_csv(q400_path)
        q400 = q400.rename(columns={"logit_score": "IC_q400_logit"})
        for src, key, nm in [("psuedfeld", "Suedfeld", "Suedfeld Exemplars (Q400)"),
                              ("jakob", "Jakob", "Jakob et al. (2021) (Q400)")]:
            sub = q400[q400["source"] == src].copy()
            if len(sub) > 0:
                results[f"{key}_q400"] = validate_corpus(
                    sub, "gt", nm, ic_col="IC_q400_logit")
    else:
        print("  validation_results_qwen3orpo400.csv not found")

    # Combined
    if all(k in results for k in ["Suedfeld_gpt", "Jakob_gpt",
                                   "Suedfeld_q400", "Jakob_q400"]):
        print(f"\n{'='*70}")
        print(f"  COMBINED SUMMARY (paper Table: validation)")
        print(f"{'='*70}")
        print(f"\n  {'Metric':<22s} {'Sue gpt':>10s} {'Sue Q400':>10s} "
              f"{'Jak gpt':>10s} {'Jak Q400':>10s}")
        print(f"  {'-'*66}")
        for key, label in [
            ("n", "N"),
            ("r", "Pearson r"),
            ("rho", "Spearman ρ"),
            ("icc", "ICC(3,1)"),
            ("kw", "Weighted κ (quad)"),
            ("mae", "MAE"),
            ("exact", "Exact agreement"),
            ("within1", "Within ±1"),
        ]:
            sg = results["Suedfeld_gpt"][key]
            sq = results["Suedfeld_q400"][key]
            jg = results["Jakob_gpt"][key]
            jq = results["Jakob_q400"][key]
            if key in ("exact", "within1"):
                print(f"  {label:<22s} {sg:>9.1%} {sq:>9.1%} {jg:>9.1%} {jq:>9.1%}")
            elif key == "n":
                print(f"  {label:<22s} {sg:>10d} {sq:>10d} {jg:>10d} {jq:>10d}")
            else:
                print(f"  {label:<22s} {sg:>10.3f} {sq:>10.3f} {jg:>10.3f} {jq:>10.3f}")


if __name__ == "__main__":
    main()
