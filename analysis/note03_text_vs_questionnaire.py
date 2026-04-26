#!/usr/bin/env python3
"""note03_text_vs_questionnaire.py — SI Note 3.

Tests whether a text-based classifier outperforms a self-report-scale
classifier at predicting evaluative-vs-compliant belief revision.

Labels:
  Restrict to "changers" (DV_BeliefChange_Specific > 0) with both
  GPT_CoT_PlausibilityRating and AIProvidedInaccurateSummary present, then
    evaluative ("good")  = plaus >= 4 AND AI-summary accurate
    compliant ("risky")  = plaus <= 2 OR  AI-summary inaccurate
  Drop participants meeting neither criterion.

Comparison design (asymmetric training, common test):
  - Text classifier trains on ALL extreme participants (n ~ 600);
    test set is restricted to the questionnaire-subset folds.
  - Self-report classifier trains on the questionnaire-subset only
    (n = 117) and is tested on the same folds.
  Both classifiers are evaluated on identical participants. The text
  classifier's larger training set is the methodological point: text
  features are universal (every participant produces text); self-report
  scales were collected only in Study 2. The fair test is whether
  text-derived signal — even with a generic lexical featuriser — beats
  self-report given each medium's natural data availability.

Classifiers:
  Text:        TF-IDF (1-2 grams, sublinear TF, 2000 features, min_df=3,
               English stopwords) + LogisticRegression (L2, C=1).
               This is the canonical recipe for sparse high-dim text
               features at moderate sample sizes.
  Self-report: IH + AOT + NFC + LogisticRegression (L2, C=1).

CV:           10-fold KFold on the questionnaire subset, averaged over
              30 random seeds (small n, so seed averaging is needed for
              a stable estimate).

Result (30-seed average):
  text  AUC = 0.675 +/- 0.011  (range 0.653 - 0.707)
  quest AUC = 0.546 +/- 0.026  (range 0.470 - 0.581)
  margin    = +0.130            (text > quest in 100% of seeds)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"

N_SEEDS = 30
N_FOLDS = 10


def load():
    raw_path = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
    if not raw_path.exists():
        from _raw_data_check import require_raw
        require_raw(
            raw_path,
            "Costello (raw publication CSV — needs GPT_CoT_PlausibilityRating "
            "and AIProvidedInaccurateSummary)",
            "https://osf.io/gdkb7/",
        )
    raw = pd.read_csv(raw_path, low_memory=False).drop_duplicates("participantId", keep="first")
    an = pd.read_csv(DATA / "analysis_data.csv")
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    text_df = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "text":          [m.get("text_initial", "") for m in meta],
    })
    alt = pd.read_csv(DATA / "alt_constructs_logitev.csv")[["participantId", "NFC_ev"]]
    extra = raw[["participantId", "GPT_CoT_PlausibilityRating", "AIProvidedInaccurateSummary"]]
    df = (an.merge(text_df, on="participantId", how="inner")
            .merge(extra, on="participantId", how="left")
            .merge(alt, on="participantId", how="left"))
    for c in ["DV_BeliefChange_Specific", "GPT_CoT_PlausibilityRating",
              "AIProvidedInaccurateSummary", "IH", "AOT", "NFC_ev"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_extreme(df):
    """Apply 04d label spec: changers with extreme plausibility / accuracy."""
    ch = df[df["DV_BeliefChange_Specific"] > 0].copy()
    ch = ch.dropna(subset=["GPT_CoT_PlausibilityRating", "AIProvidedInaccurateSummary"])
    ch["good"]  = ((ch["GPT_CoT_PlausibilityRating"] >= 4) &
                   (ch["AIProvidedInaccurateSummary"] == 0)).astype(int)
    ch["risky"] = ((ch["GPT_CoT_PlausibilityRating"] <= 2) |
                   (ch["AIProvidedInaccurateSummary"] == 1)).astype(int)
    extreme = ch[(ch["good"] == 1) | (ch["risky"] == 1)].copy()
    extreme["is_risky"] = extreme["risky"].astype(int)
    extreme = extreme[extreme["text"].str.len() > 0].reset_index(drop=True)
    return extreme


def main():
    df = load()
    extreme = build_extreme(df)
    q_sub = extreme.dropna(subset=["IH", "AOT", "NFC_ev"]).reset_index(drop=True)

    print("SI Note 3 — Text vs questionnaire classifier on evaluative-vs-compliant labels\n")
    print(f"Extreme set (changers with extreme plaus or AI-inaccuracy):  n = {len(extreme)}")
    print(f"  evaluative (good)  = {(extreme['is_risky'] == 0).sum()}")
    print(f"  compliant  (risky) = {(extreme['is_risky'] == 1).sum()}")
    print(f"Questionnaire subset (IH + AOT + NFC all present):          n = {len(q_sub)}")
    print(f"  evaluative = {(q_sub['is_risky'] == 0).sum()},  compliant = {(q_sub['is_risky'] == 1).sum()}")

    text_aucs, quest_aucs = [], []
    for seed in range(N_SEEDS):
        kf = KFold(N_FOLDS, shuffle=True, random_state=seed)
        pt = np.full(len(q_sub), np.nan)
        pq = np.full(len(q_sub), np.nan)
        for tr_q, te_q in kf.split(q_sub):
            # Text classifier: train on ALL extreme participants except the
            # current test fold's pids; test on this fold's pids.
            te_pids = set(q_sub.iloc[te_q]["participantId"].values)
            train_ex = extreme[~extreme["participantId"].isin(te_pids)]
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                  max_features=2000, min_df=3, stop_words="english")
            X_tr = vec.fit_transform(train_ex["text"])
            y_tr = train_ex["is_risky"].values
            mt = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr, y_tr)
            pt[te_q] = mt.predict_proba(vec.transform(q_sub.iloc[te_q]["text"]))[:, 1]
            # Questionnaire classifier: train on the q-subset only.
            X_q_tr = q_sub.iloc[tr_q][["IH", "AOT", "NFC_ev"]].values
            y_q_tr = q_sub.iloc[tr_q]["is_risky"].values
            mq = LogisticRegression(max_iter=2000, C=1.0).fit(X_q_tr, y_q_tr)
            pq[te_q] = mq.predict_proba(q_sub.iloc[te_q][["IH", "AOT", "NFC_ev"]].values)[:, 1]
        text_aucs.append(roc_auc_score(q_sub["is_risky"], pt))
        quest_aucs.append(roc_auc_score(q_sub["is_risky"], pq))

    t, q = np.array(text_aucs), np.array(quest_aucs)
    margin = t - q

    print(f"\n{N_SEEDS}-seed averaged results (10-fold CV, asymmetric training, common test):")
    print(f"  Text  classifier (TF-IDF + LR, train on all extreme):")
    print(f"    AUC = {t.mean():.3f} +/- {t.std():.3f}  (range {t.min():.3f} - {t.max():.3f})")
    print(f"  Quest classifier (IH + AOT + NFC + LR, train on q-subset):")
    print(f"    AUC = {q.mean():.3f} +/- {q.std():.3f}  (range {q.min():.3f} - {q.max():.3f})")
    print(f"  Margin (text - quest): {margin.mean():+.3f} +/- {margin.std():.3f}")
    print(f"  P(text > quest across seeds): {(t > q).mean():.2f}")


if __name__ == "__main__":
    main()
