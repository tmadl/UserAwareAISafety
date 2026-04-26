#!/usr/bin/env python3
"""note19_ai_side_ic.py — SI Note 19.

AI-side argument-complexity analyses for Costello: does the user-side
inverted-U survive controlling for AI-side IC, does AI-side IC act as an
additive main effect or a user-AI gap effect, are user x AI interactions
null, and does the user-side curvature persist under topic fixed effects?

Reproduces three SI tables:
  tab:aiic_survival     — user IC^2 under AI-side controls
  tab:complexity_vs_gap — additive vs gap nested test
  tab:topic_fe          — within-topic refit (user IC^2, AI IC, per-turn)

Inputs (all bundled):
  data/costello2024/costello_gpt_ic_qwenorpo400.csv
    Per-turn AI-side IC scores (logit-EV) and word counts. No GPT text.
  data/costello2024/analysis_data.csv
  data/costello2024/texts_for_scoring.jsonl
  data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv

Topic fixed-effects test additionally needs the raw Costello publication
CSV for the conspiracyTheory free-text column (used to derive the 20
TF-IDF/KMeans topic clusters); that section gracefully skips if absent.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"
RAW = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"

COVARS = ("Pre_Belief_Specific", "OpenendedResponseWordCount")


def zs(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def load_frame() -> pd.DataFrame:
    gpt = pd.read_csv(DATA / "costello_gpt_ic_qwenorpo400.csv")
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    uic = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ad = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)

    user = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "user_ic":        uic["ic_qwenorpo400_logit"].astype(float).values,
    })

    turn = ["ic_r1_logit", "ic_r2_logit", "ic_r3_logit"]
    gpt["gpt_mean"]  = gpt[turn].mean(axis=1)
    gpt["gpt_slope"] = (gpt["ic_r3_logit"] - gpt["ic_r1_logit"]) / 2.0
    gpt["gpt_sd"]    = gpt[turn].std(axis=1, ddof=0)
    gpt["gpt_concat"] = gpt["ic_concat_logit"]

    keep = ["participantId", *turn, "gpt_mean", "gpt_concat", "gpt_slope", "gpt_sd"]
    df = (user
          .merge(gpt[keep], on="participantId", how="inner")
          .merge(ad[["participantId", "DV_BeliefChange_Specific", *COVARS]],
                 on="participantId", how="inner"))
    return df.dropna(subset=["user_ic", "gpt_mean", "DV_BeliefChange_Specific",
                              *COVARS]).reset_index(drop=True)


def ols(y, X):
    return sm.OLS(y, X).fit()


def main():
    df = load_frame()
    print(f"SI Note 19 — AI-side argument complexity")
    print(f"N (treatment, complete-cases) = {len(df)}\n")

    y = df["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    cov_z = [zs(df[c].values) for c in COVARS]
    ic = zs(df["user_ic"].values); ic2 = zs(df["user_ic"].values ** 2)
    gm = zs(df["gpt_mean"].values); gs = zs(df["gpt_slope"].values)
    gsd = zs(df["gpt_sd"].values);  gc = zs(df["gpt_concat"].values)

    # Table tab:aiic_survival
    print("Table 1 — User IC^2 under AI-side IC controls:")
    print(f"  {'Added AI-side control(s)':<44}{'beta_IC^2':>10}  {'p':>9}  {'R^2':>7}")
    configs = [
        ("(none; reference)",                      [ic, ic2]),
        ("+ gpt_mean",                             [ic, ic2, gm]),
        ("+ gpt_concat",                           [ic, ic2, gc]),
        ("+ gpt_mean + slope + SD",                [ic, ic2, gm, gs, gsd]),
    ]
    for label, regs in configs:
        X = sm.add_constant(np.column_stack([*regs, *cov_z]))
        m = ols(y, X)
        print(f"  {label:<44}{m.params[2]:>+10.3f}  {m.pvalues[2]:>9.4f}  {m.rsquared:>7.4f}")

    # Table tab:complexity_vs_gap — nested additive vs gap test
    print("\nTable 2 — AI complexity main effect vs user-AI gap:")
    # A. user_ic + gpt_mean, both linear (no IC^2)
    X_A = sm.add_constant(np.column_stack([ic, gm, *cov_z]))
    m_A = ols(y, X_A)
    print(f"  A. User IC + AI IC (linear):   "
          f"beta(AI)={m_A.params[2]:+.3f}  p={m_A.pvalues[2]:.4f}")
    # B. paper spec + gpt_mean
    X_B = sm.add_constant(np.column_stack([ic, ic2, gm, *cov_z]))
    m_B = ols(y, X_B)
    print(f"  B. Paper spec + gpt_mean:      "
          f"beta(AI)={m_B.params[3]:+.3f}  p={m_B.pvalues[3]:.4f}")

    # Tertile decomposition for AI main effect
    df["u_tert"] = pd.qcut(df["user_ic"], 3, labels=["low", "mid", "high"])
    print(f"  C. User-IC tertile, beta(AI):")
    for lab, sub in df.groupby("u_tert", observed=True):
        yi = sub["DV_BeliefChange_Specific"].to_numpy(dtype=float)
        Xi = sm.add_constant(np.column_stack([zs(sub["gpt_mean"].values),
                                              *[zs(sub[c].values) for c in COVARS]]))
        mi = ols(yi, Xi)
        print(f"     {str(lab):<5}  n={len(sub):>4}  "
              f"beta={mi.params[1]:+.3f}  p={mi.pvalues[1]:.3f}")

    # Nested test: does additive nest gap (β_AI + β_user = 0)?
    # Individually z-standardised AI and user (each on their own scale),
    # so the gap restriction is β_AI + β_user = 0.
    z_u = ic; z_u2 = ic2; z_g = gm
    X_add = sm.add_constant(np.column_stack([z_u, z_u2, z_g, *cov_z]))
    m_add = ols(y, X_add)
    z_gap = z_g - z_u
    X_gap = sm.add_constant(np.column_stack([z_gap, z_u2, *cov_z]))
    m_gap = ols(y, X_gap)
    f = m_add.compare_f_test(m_gap)
    bf_add_vs_gap = float(np.exp((m_gap.bic - m_add.bic) / 2))
    print(f"\n  Nested additive vs gap (individually z-standardised AI and user):")
    print(f"    additive: beta_user={m_add.params[1]:+.3f}  beta_user^2={m_add.params[2]:+.3f}  "
          f"beta_AI={m_add.params[3]:+.3f}  R^2={m_add.rsquared:.4f}")
    print(f"    gap:      beta_gap={m_gap.params[1]:+.3f}  beta_user^2={m_gap.params[2]:+.3f}  "
          f"R^2={m_gap.rsquared:.4f}")
    print(f"    F-test (additive vs gap): F={f[0]:.2f}  p={f[1]:.4f}")
    print(f"    BF10 (additive vs gap)  = {bf_add_vs_gap:.1f}")
    print(f"    Implied user/AI ratio (gap parameterisation requires 1.00): "
          f"{m_add.params[1] / (-m_add.params[3]):.2f}")

    # User x AI interactions
    print("\nUser x AI-IC interactions (vs additive, BF reported):")
    X_lin = sm.add_constant(np.column_stack([ic, ic2, gm, ic * gm, *cov_z]))
    X_q   = sm.add_constant(np.column_stack([ic, ic2, gm, ic2 * gm, *cov_z]))
    m_lin, m_q = ols(y, X_lin), ols(y, X_q)
    bf_lin = float(np.exp((m_B.bic - m_lin.bic) / 2))
    bf_q   = float(np.exp((m_B.bic - m_q.bic)   / 2))
    print(f"  user_ic   x gpt_mean  (linear):    "
          f"beta={m_lin.params[4]:+.3f}  p={m_lin.pvalues[4]:.3f}  BF10={bf_lin:.2f}")
    print(f"  user_ic^2 x gpt_mean  (quadratic): "
          f"beta={m_q.params[4]:+.3f}  p={m_q.pvalues[4]:.3f}  BF10={bf_q:.2f}")

    # Table tab:topic_fe (needs raw conspiracyTheory text)
    if not RAW.exists():
        print("\n[Topic fixed-effects: skipped — needs raw Costello CSV for conspiracyTheory text.]")
        return
    raw = pd.read_csv(RAW, low_memory=False).drop_duplicates("participantId", keep="first")
    df_t = df.merge(raw[["participantId", "conspiracyTheory"]],
                    on="participantId", how="left").dropna(subset=["conspiracyTheory"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    tfidf = TfidfVectorizer(max_features=500, stop_words="english",
                            ngram_range=(1, 2), min_df=3)
    X = tfidf.fit_transform(df_t["conspiracyTheory"].fillna("").astype(str))
    km = KMeans(n_clusters=20, random_state=42, n_init=10).fit(X)
    df_t = df_t.assign(topic_k=km.labels_).reset_index(drop=True)
    tdum = pd.get_dummies(df_t["topic_k"], drop_first=True).astype(float).values

    print(f"\nTable 3 — Topic fixed-effects (TF-IDF + KMeans k=20, n={len(df_t)}):")
    y_t = df_t["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    u_t = zs(df_t["user_ic"].values); u2_t = zs(df_t["user_ic"].values ** 2)
    cov_t = [zs(df_t[c].values) for c in COVARS]

    print(f"  {'Predictor':<28}  {'no FE':>10}  {'+ topic FE':>10}")
    # User IC^2
    X_no = sm.add_constant(np.column_stack([u_t, u2_t, *cov_t]))
    X_fe = sm.add_constant(np.column_stack([u_t, u2_t, *cov_t, tdum]))
    print(f"  {'User IC^2 (paper spec)':<28}  "
          f"{ols(y_t, X_no).params[2]:>+10.3f}  {ols(y_t, X_fe).params[2]:>+10.3f}")
    # gpt_mean (paper spec adds gm to spec)
    gm_t = zs(df_t["gpt_mean"].values)
    X_no = sm.add_constant(np.column_stack([gm_t, u_t, u2_t, *cov_t]))
    X_fe = sm.add_constant(np.column_stack([gm_t, u_t, u2_t, *cov_t, tdum]))
    print(f"  {'gpt_mean (paper spec)':<28}  "
          f"{ols(y_t, X_no).params[1]:>+10.3f}  {ols(y_t, X_fe).params[1]:>+10.3f}")
    # Per-turn IC (univariate paper-spec adds)
    for lab, col in [("ic_r1 (turn-1)", "ic_r1_logit"),
                      ("ic_r2 (turn-2)", "ic_r2_logit"),
                      ("ic_r3 (turn-3)", "ic_r3_logit")]:
        pred = zs(df_t[col].values)
        X_no = sm.add_constant(np.column_stack([pred, u_t, u2_t, *cov_t]))
        X_fe = sm.add_constant(np.column_stack([pred, u_t, u2_t, *cov_t, tdum]))
        print(f"  {lab:<28}  "
              f"{ols(y_t, X_no).params[1]:>+10.3f}  {ols(y_t, X_fe).params[1]:>+10.3f}")


if __name__ == "__main__":
    main()
