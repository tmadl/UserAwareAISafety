#!/usr/bin/env python3
"""
10_topic_fixed_effects.py — Within-topic Costello inverted-U.

Reviewer Overall #1: the aggregate IC² curvature (β = -15.17) is partly
driven by which conspiracy topics low- vs. high-IC users gravitate toward.
Controlling for topic (TF-IDF + KMeans clustering of the free-text
conspiracyTheory field, k = 20), the within-topic quadratic shrinks but
survives (β ≈ -8.76, p ≈ .008).

Inputs:
  data/costello2024/analysis_data.csv
  data/costello2024/all_complexity_scores.csv
  data/costello2024/orpo_ic_scores.csv
  data/costello2024/Data 8.28.24/AllDataForPublication.PPI.8.28.24.csv
"""

import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q400 = ROOT / "data" / "ic_qwen3orpo400"

K_TOPICS = 20
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 3
SEED = 42


def zs(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def load_data():
    """Merge primary IC (Qwen3-ORPO-400 logit-EV) + cross-scorer (gpt-4.1-mini)
    with belief change and conspiracyTheory text.

    Primary scorer: ic_qwenorpo400_logit, scored on text_initial
    (pre-treatment belief statement), merged via texts_for_scoring.jsonl order.
    """
    an = pd.read_csv(DATA / "analysis_data.csv")

    # gpt-4.1-mini cross-scorer
    ic_gpt = pd.read_csv(DATA / "all_complexity_scores.csv")[
        ["participantId", "IC_openai"]]

    # Primary Q400 logit-EV (row-indexed to jsonl)
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q400 = pd.read_csv(DATA_Q400 /
                       "costello_texts_for_scoring_initial_qwenorpo400.csv")
    assert len(meta) == len(q400)
    ic_q400 = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "ic_q400_logit": q400["ic_qwenorpo400_logit"].astype(float).values,
    })

    df = an.merge(ic_gpt, on="participantId", how="inner")
    df = df.merge(ic_q400, on="participantId", how="left")

    # Pull conspiracyTheory text (one row per participant)
    from _raw_data_check import require_raw
    _raw_path = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
    require_raw(_raw_path, "Costello", "https://osf.io/gdkb7/")
    raw = pd.read_csv(_raw_path, low_memory=False)
    raw = raw.drop_duplicates(subset="participantId", keep="first")
    df = df.merge(raw[["participantId", "conspiracyTheory"]], on="participantId",
                  how="left")

    # Numerics
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_topic_clusters(df, k=K_TOPICS, seed=SEED):
    """TF-IDF + KMeans on conspiracyTheory text."""
    texts = df["conspiracyTheory"].fillna("").astype(str).values
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, stop_words="english",
        ngram_range=(1, 2), min_df=TFIDF_MIN_DF)
    X = tfidf.fit_transform(texts)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    return km.labels_


def analysis_topic_fe(df, ic_col, label):
    print(f"\n{'='*70}")
    print(f"  WITHIN-TOPIC IC² — {label}")
    print(f"{'='*70}")

    dv = "DV_BeliefChange_Specific"
    covs = ["Pre_Belief_Specific", "OpenendedResponseWordCount"]

    sub = df.dropna(subset=[dv, ic_col, "conspiracyTheory"] + covs).copy()
    sub["topic_k"] = build_topic_clusters(sub, k=K_TOPICS, seed=SEED)
    print(f"  N = {len(sub)} (k = {K_TOPICS} topic clusters)")

    y = sub[dv].values
    ic = zs(sub[ic_col].values)
    ic2 = zs(sub[ic_col].values ** 2)
    pre = zs(sub[covs[0]].values)
    wc = zs(sub[covs[1]].values)

    tdum = pd.get_dummies(sub["topic_k"], drop_first=True).astype(float).values

    # Without topic FE
    X_no = sm.add_constant(np.column_stack([ic, ic2, pre, wc]))
    m_no = sm.OLS(y, X_no).fit()

    # With topic FE
    X_fe = sm.add_constant(np.column_stack([ic, ic2, pre, wc, tdum]))
    m_fe = sm.OLS(y, X_fe).fit()

    print(f"\n  Without topic FE:")
    print(f"    β_IC  = {m_no.params[1]:+.3f} (p = {fmt_p(m_no.pvalues[1])})")
    print(f"    β_IC² = {m_no.params[2]:+.3f} (p = {fmt_p(m_no.pvalues[2])})")
    print(f"    R² = {m_no.rsquared:.4f}")

    print(f"\n  With topic FE (k = {K_TOPICS}):")
    print(f"    β_IC  = {m_fe.params[1]:+.3f} (p = {fmt_p(m_fe.pvalues[1])})")
    print(f"    β_IC² = {m_fe.params[2]:+.3f} (p = {fmt_p(m_fe.pvalues[2])})")
    print(f"    R² = {m_fe.rsquared:.4f}")

    retention = m_fe.params[2] / m_no.params[2] * 100
    print(f"\n  β_IC² retention under topic FE: {retention:.1f}% "
          f"({m_no.params[2]:+.2f} → {m_fe.params[2]:+.2f})")

    return {
        "label": label, "n": len(sub),
        "beta_ic2_no": m_no.params[2], "p_ic2_no": m_no.pvalues[2],
        "beta_ic2_fe": m_fe.params[2], "p_ic2_fe": m_fe.pvalues[2],
        "retention": retention,
    }


def main():
    df = load_data()
    print(f"Loaded {len(df)} participants")

    res_orpo = analysis_topic_fe(df, "ic_q400_logit", "Primary (Q400 logit-EV)")
    res_gpt = analysis_topic_fe(df, "IC_openai", "Cross-scorer (gpt-4.1-mini)")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Scorer':<30s} {'β_IC² no FE':>13s} {'β_IC² +FE':>13s} {'retention':>10s}")
    print(f"  {'-'*68}")
    for r in [res_orpo, res_gpt]:
        print(f"  {r['label']:<30s} {r['beta_ic2_no']:>+13.3f} "
              f"{r['beta_ic2_fe']:>+13.3f} {r['retention']:>9.1f}%")


if __name__ == "__main__":
    main()
