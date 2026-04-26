#!/usr/bin/env python3
"""note22_loso_enrichment.py — SI Note 22.

Out-of-sample enrichment via leave-one-study-out (LOSO) cross-validation
on the pooled three-study Costello corpus. For each study k in {1, 2, 3},
the IC bottom-quintile threshold is fixed on the other two studies (the
training pool) and applied to study k (the held-out pool); the resulting
held-out participants form a pooled LOSO set of size N = 1,782 in which
no participant contributed to the threshold used to classify them.

Reproduces SI Note 22 headline numbers:
  - large-change (>=20 pt) retention under LOSO     ~ 86.1%
  - adverse-movement (<=-5 pt) exclusion under LOSO ~ 24.9%
  - AUC ~ 0.69 for predicting adverse movement under LOSO from a logistic
    model with IC + IC^2 + pre-belief + word-count predictors trained on
    the held-out fold's complement (univariate -IC alone gives ~ 0.53,
    reported here for completeness)

Inputs:
  data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv
  data/costello2024/analysis_data.csv
  data/costello2024/Data 8.28.24/AllDataForPublication.PPI.8.28.24.csv
    (raw Costello — for StudyNumber column)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"
RAW = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"

LARGE_THRESHOLD    = 20    # large-change DV >= +20 pts
ADVERSE_THRESHOLD  = -5    # adverse movement DV <= -5 pts
QUINTILE           = 0.20  # bottom-quintile cut


def main():
    if not RAW.exists():
        from _raw_data_check import require_raw
        require_raw(RAW, "Costello (raw publication CSV — needs StudyNumber)",
                    "https://osf.io/gdkb7/")

    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    ic = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ad = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)
    raw = pd.read_csv(RAW, low_memory=False).drop_duplicates("participantId", keep="first")

    base = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC":             ic["ic_qwenorpo400_logit"].astype(float).values,
    })
    df = (base.merge(ad[["participantId", "DV_BeliefChange_Specific",
                          "Pre_Belief_Specific", "OpenendedResponseWordCount"]],
                     on="participantId", how="inner")
              .merge(raw[["participantId", "StudyNumber"]], on="participantId", how="left"))
    df = df.dropna(subset=["DV_BeliefChange_Specific", "IC",
                            "Pre_Belief_Specific", "OpenendedResponseWordCount",
                            "StudyNumber"]).reset_index(drop=True)
    df["StudyNumber"] = df["StudyNumber"].astype(int)

    print("SI Note 22 — Out-of-sample enrichment (leave-one-study-out)")
    print(f"Working N = {len(df)}, studies = {sorted(df['StudyNumber'].unique())}\n")

    # Per-fold quintile cut + retention/exclusion
    fold_rows = []
    pool = []
    for study in sorted(df["StudyNumber"].unique()):
        train = df[df["StudyNumber"] != study]
        test  = df[df["StudyNumber"] == study].copy()
        q1_cut = train["IC"].quantile(QUINTILE)
        test["excluded"] = (test["IC"] <= q1_cut).astype(int)
        test["large"]    = (test["DV_BeliefChange_Specific"] >= LARGE_THRESHOLD).astype(int)
        test["adverse"]  = (test["DV_BeliefChange_Specific"] <= ADVERSE_THRESHOLD).astype(int)
        n_large_t   = test["large"].sum()
        n_adverse_t = test["adverse"].sum()
        kept_large  = test.loc[test["excluded"] == 0, "large"].sum()
        excl_adv    = test.loc[test["excluded"] == 1, "adverse"].sum()
        fold_rows.append({
            "held_out_study": study,
            "n_train": len(train), "n_test": len(test),
            "q1_cut_from_train": float(q1_cut),
            "large_retained_pct":   100 * kept_large / max(n_large_t, 1),
            "adverse_excluded_pct": 100 * excl_adv  / max(n_adverse_t, 1),
        })
        pool.append(test)

    fold_df = pd.DataFrame(fold_rows)
    print("Per-fold:")
    print(fold_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    pooled = pd.concat(pool, ignore_index=True)
    n_large_tot   = pooled["large"].sum()
    n_adv_tot     = pooled["adverse"].sum()
    kept_large    = pooled.loc[pooled["excluded"] == 0, "large"].sum()
    excl_adverse  = pooled.loc[pooled["excluded"] == 1, "adverse"].sum()

    print(f"\nPooled LOSO ({len(pooled)} held-out participants):")
    print(f"  large-change ({LARGE_THRESHOLD:+d}+ pt) retained outside excluded set: "
          f"{kept_large}/{n_large_tot} = {100 * kept_large / n_large_tot:.1f}%")
    print(f"  adverse-movement ({ADVERSE_THRESHOLD:+d} or worse) excluded:           "
          f"{excl_adverse}/{n_adv_tot} = {100 * excl_adverse / n_adv_tot:.1f}%")

    # Univariate AUC: rank by -IC alone
    auc_uni = roc_auc_score(pooled["adverse"].values, -pooled["IC"].values)
    print(f"\nUnivariate AUC (rank by -IC alone, no covariates): {auc_uni:.3f}")

    # Multivariate LOSO AUC: logistic regression IC + IC^2 + pre-belief + word count
    auc_rows = []
    for study in sorted(df["StudyNumber"].unique()):
        train = df[df["StudyNumber"] != study]
        test  = df[df["StudyNumber"] == study]
        y_tr = (train["DV_BeliefChange_Specific"] <= ADVERSE_THRESHOLD).astype(int).values
        y_te = (test ["DV_BeliefChange_Specific"] <= ADVERSE_THRESHOLD).astype(int).values
        X_tr = np.column_stack([train["IC"].values, train["IC"].values ** 2,
                                train["Pre_Belief_Specific"].values,
                                train["OpenendedResponseWordCount"].values])
        X_te = np.column_stack([test ["IC"].values, test ["IC"].values ** 2,
                                test ["Pre_Belief_Specific"].values,
                                test ["OpenendedResponseWordCount"].values])
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        lr = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
        auc_rows.append({"held_out_study": study,
                         "n_test": len(y_te), "n_adverse_test": int(y_te.sum()),
                         "auc_adverse": roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])})
    auc_df = pd.DataFrame(auc_rows)
    print(f"\nMultivariate LOSO AUC (IC + IC^2 + pre-belief + word count, logistic):")
    print(auc_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  mean across folds: AUC = {auc_df['auc_adverse'].mean():.3f}")


if __name__ == "__main__":
    main()
