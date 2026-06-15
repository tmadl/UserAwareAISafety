#!/usr/bin/env python3
"""note22c_auc_decomposition_ih.py — extends note22b's adverse-movement AUC
decomposition with text-scored Intellectual Humility (IH) features.

IH source (pre-release, outside this repository):
  /mnt/workvm/UserAwareAISafety/_orpo/ih2/costello_texts_for_scoring_initial_ih_scores.csv
    - pre-treatment text IH (3-seed ensemble EV, column ens_ev_mean); no
      participantId, so rows are joined to participantId via exact match of
      `Paragraph` against `text_initial` in costello_combined_ih_scores.csv
      (1,782 of 1,783 match uniquely; no duplicate texts).

Grid: feature sets {IH alone, IH+IH^2, IC+IH (+squares), full+IH} added to
the note22b baseline rows, x {logit, LightGBM} x {LOSO, pooled 5-fold,
within-study 5-fold}. Primary IC = IC_q400_logit; IH = ens_ev_mean.

Output: prints the grid; writes note22c_auc_decomposition_ih_output.txt.
"""
import importlib.util
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
    return LogisticRegression(max_iter=5000, C=1e6)


IH_DIR = Path("/mnt/workvm/UserAwareAISafety/_orpo/ih2")
IC = "IC_q400_logit"
PB = "Pre_Belief_Specific"
WC = "OpenendedResponseWordCount"
DV = "DV_BeliefChange_Specific"
IH = "IH_pre"


def load_ih():
    ini = pd.read_csv(IH_DIR / "costello_texts_for_scoring_initial_ih_scores.csv")
    comb = pd.read_csv(IH_DIR / "costello_combined_ih_scores.csv")
    ini["key"] = ini["Paragraph"].astype(str).str.strip()
    comb["key"] = comb["text_initial"].astype(str).str.strip()
    m = ini.merge(comb[["participantId", "key"]].drop_duplicates("key"),
                  on="key", how="inner")
    return m[["participantId", "ens_ev_mean"]].rename(
        columns={"ens_ev_mean": IH})


FEATURE_SETS = {
    "IC alone":              lambda d: d[[IC]].values,
    "IC + IC^2":             lambda d: np.column_stack([d[IC], d[IC] ** 2]),
    "IH alone":              lambda d: d[[IH]].values,
    "IH + IH^2":             lambda d: np.column_stack([d[IH], d[IH] ** 2]),
    "IC+IH (+squares)":      lambda d: np.column_stack(
        [d[IC], d[IC] ** 2, d[IH], d[IH] ** 2]),
    "pre-belief alone":      lambda d: d[[PB]].values,
    "full (IC,IC2,PB,WC)":   lambda d: np.column_stack(
        [d[IC], d[IC] ** 2, d[PB], d[WC]]),
    "full + IH,IH^2":        lambda d: np.column_stack(
        [d[IC], d[IC] ** 2, d[IH], d[IH] ** 2, d[PB], d[WC]]),
}

MODELS = {"logit": make_logit, GBM_NAME: make_gbm}


def fold_auc(factory, build, tr, te):
    m = factory()
    m.fit(build(tr), tr["adverse"].values)
    return roc_auc_score(te["adverse"].values, m.predict_proba(build(te))[:, 1])


def run():
    df = c01.load_data()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.merge(load_ih(), on="participantId", how="inner")
    df = df.dropna(subset=[IC, IH, DV, PB, WC, "StudyNumber"]).copy()
    df["adverse"] = (df[DV] <= -5).astype(int)
    print(f"N = {len(df)}, adverse = {df['adverse'].sum()} "
          f"({df['adverse'].mean() * 100:.1f}%); IC = {IC}; IH = ens_ev_mean "
          f"(pre-treatment text, 3-seed ensemble); GBM = {GBM_NAME}")
    print(f"IH descriptives: mean = {df[IH].mean():.2f}, SD = {df[IH].std():.2f}, "
          f"range = [{df[IH].min():.2f}, {df[IH].max():.2f}]; "
          f"r(IC, IH) = {df[IC].corr(df[IH]):.3f}")
    print()
    header = (f"{'features':22s} {'model':18s} {'LOSO':>6s} {'pooled5f':>9s} "
              f"{'within-study 5f (S1/S2/S3)':>28s}")
    print(header)
    print("-" * len(header))
    for fname, build in FEATURE_SETS.items():
        for mname, factory in MODELS.items():
            loso = np.mean([
                fold_auc(factory, build,
                         df[df["StudyNumber"] != s], df[df["StudyNumber"] == s])
                for s in (1, 2, 3)])
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            pooled = np.mean([
                fold_auc(factory, build, df.iloc[tr_i], df.iloc[te_i])
                for tr_i, te_i in skf.split(df, df["adverse"].values)])
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


if __name__ == "__main__":
    buf = io.StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    out = HERE / "note22c_auc_decomposition_ih_output.txt"
    out.write_text(text)
    sys.stdout.write(f"\nWrote {out}\n")
