#!/usr/bin/env python3
"""note22b_auc_decomposition.py — AUC decomposition for adverse-movement
discrimination (SI Note 22, individual-level prediction passage).

Question: how much of the paper-spec LOSO AUC = 0.69 for adverse movement
(DV_BeliefChange_Specific <= -5) is attributable to IC, as opposed to the
covariates (pre-treatment belief strength, word count)?

Grid: feature sets {IC alone, IC+IC^2, pre-belief alone, pre-belief+WC,
full paper spec} x models {logistic regression, LightGBM (shape-free)} x
CV schemes {leave-one-study-out, pooled stratified 5-fold, within-study
stratified 5-fold}. All on the primary scorer (Qwen3.5-ORPO-400 logit-EV,
column IC_q400_logit). AUC = area under ROC, mean across folds.

Conclusion encoded in the SI text: the 0.69 is carried almost entirely by
pre-belief (~0.70 alone); IC-only models sit at ~0.46-0.56 under every
functional form and CV scheme, i.e. IC carries no individual-level
discrimination of adverse movement. The IC contribution is the
population-level moderation pattern, not individual risk ranking.

Output: prints the grid; also writes note22b_auc_decomposition_output.txt
next to this script.

Requires: scikit-learn; lightgbm (falls back to sklearn
HistGradientBoostingClassifier if lightgbm is unavailable).
"""
import importlib.util
import io
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c01", HERE / "01_costello_analysis.py")
c01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c01)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from lightgbm import LGBMClassifier

    def make_gbm():
        return LGBMClassifier(n_estimators=200, num_leaves=7, learning_rate=0.05,
                              min_child_samples=30, verbose=-1, random_state=0)

    GBM_NAME = "LightGBM"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    def make_gbm():
        return HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=7,
                                              learning_rate=0.05,
                                              min_samples_leaf=30, random_state=0)

    GBM_NAME = "HistGBM(sklearn)"


def make_logit():
    # C large -> effectively unregularised, matching the statsmodels Logit
    # used for the published AUC = 0.69.
    return LogisticRegression(max_iter=5000, C=1e6)


IC = "IC_q400_logit"
PB = "Pre_Belief_Specific"
WC = "OpenendedResponseWordCount"
DV = "DV_BeliefChange_Specific"

FEATURE_SETS = {
    "IC alone":            lambda d: d[[IC]].values,
    "IC + IC^2":           lambda d: np.column_stack([d[IC].values, d[IC].values ** 2]),
    "pre-belief alone":    lambda d: d[[PB]].values,
    "pre-belief + WC":     lambda d: d[[PB, WC]].values,
    "full (IC,IC2,PB,WC)": lambda d: np.column_stack(
        [d[IC].values, d[IC].values ** 2, d[PB].values, d[WC].values]),
}

MODELS = {"logit": make_logit, GBM_NAME: make_gbm}


def fold_auc(model_factory, build, tr, te):
    m = model_factory()
    m.fit(build(tr), tr["adverse"].values)
    p = m.predict_proba(build(te))[:, 1]
    return roc_auc_score(te["adverse"].values, p)


def run():
    df = c01.load_data()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(subset=[IC, DV, PB, WC, "StudyNumber"]).copy()
    df["adverse"] = (df[DV] <= -5).astype(int)
    print(f"N = {len(df)}, adverse = {df['adverse'].sum()} "
          f"({df['adverse'].mean() * 100:.1f}%); scorer column = {IC}; "
          f"GBM = {GBM_NAME}")
    print()

    header = (f"{'features':22s} {'model':18s} {'LOSO':>6s} {'pooled5f':>9s} "
              f"{'within-study 5f (S1/S2/S3)':>28s}")
    print(header)
    print("-" * len(header))

    for fname, build in FEATURE_SETS.items():
        for mname, factory in MODELS.items():
            # LOSO
            loso = np.mean([
                fold_auc(factory, build,
                         df[df["StudyNumber"] != s], df[df["StudyNumber"] == s])
                for s in (1, 2, 3)])
            # Pooled stratified 5-fold (individuals held out, studies mixed)
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            pooled = np.mean([
                fold_auc(factory, build, df.iloc[tr_i], df.iloc[te_i])
                for tr_i, te_i in skf.split(df, df["adverse"].values)])
            # Within-study stratified 5-fold
            within = []
            for s in (1, 2, 3):
                sub = df[df["StudyNumber"] == s]
                skf_s = StratifiedKFold(5, shuffle=True, random_state=42)
                within.append(np.mean([
                    fold_auc(factory, build, sub.iloc[tr_i], sub.iloc[te_i])
                    for tr_i, te_i in skf_s.split(sub, sub["adverse"].values)]))
            within_str = "/".join(f"{a:.3f}" for a in within)
            print(f"{fname:22s} {mname:18s} {loso:6.3f} {pooled:9.3f} "
                  f"{within_str:>28s}")

    # Reference: published univariate monotone -IC ranking (no model fit)
    aucs = [roc_auc_score(df[df['StudyNumber'] == s]['adverse'].values,
                          -df[df['StudyNumber'] == s][IC].values)
            for s in (1, 2, 3)]
    print()
    print(f"Reference: monotone -IC ranking (published 0.53): "
          f"per-study mean = {np.mean(aucs):.3f}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    out = HERE / "note22b_auc_decomposition_output.txt"
    out.write_text(text)
    sys.stdout.write(f"\nWrote {out}\n")
