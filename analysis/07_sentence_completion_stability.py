#!/usr/bin/env python3
"""
07_sentence_completion_stability.py — IC cross-content stability.

Tests whether IC scores are stable across thematically diverse sentence
completion stems, addressing the reviewer concern that IC scored from
task-relevant text may absorb topic-specific content rather than capturing
cross-context cognitive style.

Analysis:
  1. Cronbach's alpha across 36 stems
  2. Average inter-stem correlation
  3. Split-half reliability (odd/even stems)
  4. Thematic cluster analysis (4 clusters)
  5. Person-level IC from stems vs. overall

Reads:  data/sentence_completions/ic_scores_by_stem.csv
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp
from itertools import combinations

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sentence_completions"

STEMS = [
    "Raising a family",
    "When I'm criticized",
    "Change is",
    "A man's job",
    "Being with other people",
    "The thing I like about myself is",
    "My mother and I",
    "What gets me into trouble is",
    "Education",
    "When people are helpless",
    "Women are lucky because",
    "A good boss",
    "A girl has a right to",
    "The past",
    "When they talked about sex, I",
    "I feel sorry",
    "When they avoided me",
    "Rules are",
    "Crime and delinquency could be halted if",
    "Men are lucky because",
    "I just can't stand people who",
    "At times s/he worried about",
    "I am",
    "If I had more money",
    "My main problem is",
    "When I get mad",
    "People who step out of line at work",
    "A husband has a right to",
    "If my mother",
    "If I were in charge",
    "My father",
    "If I can't get what I want",
    "When I am nervous",
    "For a woman a career is",
    "My conscience bothers me if",
    "Sometimes s/he wished that",
    "A true friend",
    "At my worst",
    "When a child will not join in group activities",
]

# Thematic clusters: stem indices (key47 values) grouped by content domain.
# Mapping from stem text to key47 index for reference.
THEMATIC_CLUSTERS = {
    "Self, Values & Direction": [5, 13, 22, 23, 24, 34, 35],
    # The thing I like about myself is, The past, I am, If I had more money,
    # My main problem is, My conscience bothers me if, Sometimes s/he wished that

    "Relationships & Belonging": [0, 4, 6, 15, 16, 27, 28, 30],
    # Raising a family, Being with other people, My mother and I, I feel sorry,
    # When they avoided me, A husband has a right to, If my mother, My father

    "Work, Roles & Authority": [3, 8, 10, 11, 12, 17, 18, 19, 26, 29, 33],
    # A man's job, Education, Women are lucky because, A good boss,
    # A girl has a right to, Rules are, Crime and delinquency could be halted if,
    # Men are lucky because, People who step out of line at work,
    # If I were in charge, For a woman a career is

    "Emotion, Stress & Coping": [1, 2, 7, 9, 14, 20, 21, 25, 31, 32],
    # When I'm criticized, Change is, What gets me into trouble is,
    # When people are helpless, When they talked about sex I,
    # I just can't stand people who, At times s/he worried about,
    # When I get mad, If I can't get what I want, When I am nervous
}

# Stems not assigned to clusters (extras beyond 36)
# key47=36 "A true friend", 37 "At my worst", 38 "When a child will not join..."
# These are included in stemwise analysis but not in thematic clusters.


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def cronbach_alpha(item_scores):
    """Cronbach's alpha from a person × item matrix (numpy array).
    Rows = persons, columns = items. NaN-aware."""
    # Drop persons with any NaN
    mask = ~np.isnan(item_scores).any(axis=1)
    X = item_scores[mask]
    n_items = X.shape[1]
    if n_items < 2 or X.shape[0] < 10:
        return np.nan, X.shape[0]

    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)

    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha, X.shape[0]


def spearman_brown(r, n_items_ratio=2):
    """Spearman-Brown prophecy formula."""
    return (n_items_ratio * r) / (1 + (n_items_ratio - 1) * r)


def load_data():
    path = DATA / "ic_scores_by_stem.csv"
    if not path.exists():
        from _raw_data_check import require_raw
        require_raw(path, "Cook-Greuter / Madl sentence-completion (N = 887)",
                    "see paper SI Note 4 for source")
    df = pd.read_csv(path)
    df["IC_openai"] = pd.to_numeric(df["IC_openai"], errors="coerce")
    df["key47"] = df["key47"].astype(int)
    return df


# ── Analysis 1: Stemwise Reliability ────────────────────────────────────

def analysis_stemwise_reliability(df):
    print(f"\n{'='*70}")
    print(f"  STEMWISE RELIABILITY (CRONBACH'S ALPHA)")
    print(f"{'='*70}")

    # Pivot to person × stem matrix
    pivot = df.pivot_table(index="fn", columns="key47", values="IC_openai")
    print(f"\n  Persons: {pivot.shape[0]}, Stems: {pivot.shape[1]}")

    # Completeness
    complete = pivot.dropna()
    print(f"  Persons with all {pivot.shape[1]} stems: {len(complete)}")

    # Cronbach's alpha
    alpha, n_used = cronbach_alpha(pivot.values)
    print(f"\n  Cronbach's α = {alpha:.3f} (N = {n_used})")

    # Average inter-stem correlation
    corr_matrix = pivot.corr()
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    upper_vals = corr_matrix.values[mask]
    upper_vals = upper_vals[~np.isnan(upper_vals)]
    print(f"\n  Average inter-stem r = {np.mean(upper_vals):.3f} "
          f"(SD = {np.std(upper_vals):.3f}, range = [{np.min(upper_vals):.3f}, {np.max(upper_vals):.3f}])")
    print(f"  Median inter-stem r = {np.median(upper_vals):.3f}")
    print(f"  Number of stem pairs: {len(upper_vals)}")

    # Distribution of per-stem means and SDs
    stem_means = pivot.mean()
    stem_sds = pivot.std()
    print(f"\n  Per-stem IC: mean of means = {stem_means.mean():.2f}, "
          f"SD of means = {stem_means.std():.2f}")
    print(f"  Per-stem IC: mean of SDs = {stem_sds.mean():.2f}")

    return pivot, alpha, np.mean(upper_vals)


# ── Analysis 2: Split-Half Reliability ──────────────────────────────────

def analysis_split_half(pivot):
    print(f"\n{'='*70}")
    print(f"  SPLIT-HALF RELIABILITY")
    print(f"{'='*70}")

    stems = sorted(pivot.columns)

    # Odd/even split
    odd_stems = [s for i, s in enumerate(stems) if i % 2 == 0]
    even_stems = [s for i, s in enumerate(stems) if i % 2 == 1]

    half1 = pivot[odd_stems].mean(axis=1)
    half2 = pivot[even_stems].mean(axis=1)

    valid = half1.notna() & half2.notna()
    r, p = sp.pearsonr(half1[valid], half2[valid])
    sb = spearman_brown(r)

    print(f"\n  Odd/even split ({len(odd_stems)} vs {len(even_stems)} stems):")
    print(f"    r = {r:.3f}, p = {fmt_p(p)}, N = {valid.sum()}")
    print(f"    Spearman-Brown corrected = {sb:.3f}")

    # Random split-halves (100 iterations)
    rng = np.random.default_rng(42)
    rs = []
    for _ in range(100):
        perm = rng.permutation(stems)
        h1 = pivot[perm[:len(perm)//2]].mean(axis=1)
        h2 = pivot[perm[len(perm)//2:]].mean(axis=1)
        v = h1.notna() & h2.notna()
        if v.sum() > 10:
            rs.append(sp.pearsonr(h1[v], h2[v])[0])

    print(f"\n  Random split-halves (100 iterations):")
    print(f"    Mean r = {np.mean(rs):.3f} (SD = {np.std(rs):.3f})")
    print(f"    Mean Spearman-Brown = {spearman_brown(np.mean(rs)):.3f}")

    return r, sb


# ── Analysis 3: Per-Stem Statistics ─────────────────────────────────────

def analysis_per_stem(df):
    print(f"\n{'='*70}")
    print(f"  PER-STEM IC STATISTICS")
    print(f"{'='*70}")

    print(f"\n  {'Key':>4s}  {'Stem':<45s} {'N':>5s} {'Mean':>6s} {'SD':>6s}")
    print(f"  {'-'*72}")

    stem_stats = []
    for key in sorted(df["key47"].unique()):
        sub = df[df["key47"] == key].dropna(subset=["IC_openai"])
        stem_label = STEMS[key] if key < len(STEMS) else f"stem_{key}"
        stem_label = stem_label[:45]
        m = sub["IC_openai"].mean()
        s = sub["IC_openai"].std()
        print(f"  {key:>4d}  {stem_label:<45s} {len(sub):>5d} {m:>6.2f} {s:>6.2f}")
        stem_stats.append({"key": key, "stem": stem_label, "n": len(sub),
                           "mean": m, "sd": s})

    return stem_stats


# ── Analysis 4: Person-Level Aggregate ──────────────────────────────────

def analysis_person_level(df):
    print(f"\n{'='*70}")
    print(f"  PERSON-LEVEL IC AGGREGATE")
    print(f"{'='*70}")

    person_ic = df.groupby("fn")["IC_openai"].agg(["mean", "std", "count"])
    person_ic.columns = ["IC_mean", "IC_sd", "n_stems"]

    print(f"\n  Persons: {len(person_ic)}")
    print(f"  Mean IC (across persons): {person_ic['IC_mean'].mean():.3f}")
    print(f"  SD of person means: {person_ic['IC_mean'].std():.3f}")
    print(f"  Mean within-person SD: {person_ic['IC_sd'].mean():.3f}")

    # ICC(1): between-person variance / total variance
    between_var = person_ic["IC_mean"].var()
    within_var = person_ic["IC_sd"].pow(2).mean()
    icc1 = between_var / (between_var + within_var)
    print(f"\n  Between-person variance: {between_var:.3f}")
    print(f"  Mean within-person variance: {within_var:.3f}")
    print(f"  ICC(1) ≈ {icc1:.3f}")

    return person_ic


# ── Analysis 5: Thematic Cluster Analysis ───────────────────────────────

def analysis_thematic_clusters(df):
    print(f"\n{'='*70}")
    print(f"  THEMATIC CLUSTER ANALYSIS")
    print(f"{'='*70}")

    cluster_names = list(THEMATIC_CLUSTERS.keys())

    # Compute person-level mean IC within each cluster
    cluster_scores = {}
    for cname, keys in THEMATIC_CLUSTERS.items():
        sub = df[df["key47"].isin(keys)].groupby("fn")["IC_openai"].mean()
        cluster_scores[cname] = sub
        n_stems = len(keys)
        print(f"\n  {cname} ({n_stems} stems):")
        print(f"    Persons with scores: {sub.notna().sum()}")
        print(f"    Mean IC: {sub.mean():.3f}, SD: {sub.std():.3f}")

    # Pairwise correlations between clusters
    print(f"\n  Pairwise cluster correlations:")
    print(f"  {'Cluster A':<30s} {'Cluster B':<30s} {'r':>6s} {'p':>8s} {'N':>5s}")
    print(f"  {'-'*82}")

    rs = []
    for (a, b) in combinations(cluster_names, 2):
        merged = pd.DataFrame({a: cluster_scores[a], b: cluster_scores[b]}).dropna()
        if len(merged) < 10:
            continue
        r, p = sp.pearsonr(merged[a], merged[b])
        rs.append(r)
        print(f"  {a:<30s} {b:<30s} {r:>6.3f} {fmt_p(p):>8s} {len(merged):>5d}")

    print(f"\n  Average pairwise cluster r = {np.mean(rs):.3f} (SD = {np.std(rs):.3f})")

    # Cronbach's alpha across 4 cluster scores
    cluster_df = pd.DataFrame(cluster_scores)
    cluster_matrix = cluster_df.values
    alpha_c, n_c = cronbach_alpha(cluster_matrix)
    print(f"  Cronbach's α (4 clusters) = {alpha_c:.3f} (N = {n_c})")

    return rs, alpha_c


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading scored sentence completions...")
    df = load_data()
    print(f"  Total rows: {len(df)}, valid IC: {df['IC_openai'].notna().sum()}")

    pivot, alpha, mean_r = analysis_stemwise_reliability(df)
    r_half, sb = analysis_split_half(pivot)
    analysis_per_stem(df)
    person_ic = analysis_person_level(df)
    cluster_rs, cluster_alpha = analysis_thematic_clusters(df)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Stemwise:")
    print(f"    Cronbach's α          = {alpha:.3f}")
    print(f"    Average inter-stem r  = {mean_r:.3f}")
    print(f"    Split-half r          = {r_half:.3f}")
    print(f"    Spearman-Brown        = {sb:.3f}")
    print(f"  Thematic clusters:")
    print(f"    Average pairwise r    = {np.mean(cluster_rs):.3f}")
    print(f"    Cronbach's α (4 cl.)  = {cluster_alpha:.3f}")
    print(f"  Sample:")
    print(f"    Persons               = {df['fn'].nunique()}")
    print(f"    Stems                 = {df['key47'].nunique()}")


if __name__ == "__main__":
    main()
