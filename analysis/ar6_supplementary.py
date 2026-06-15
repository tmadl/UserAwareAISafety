#!/usr/bin/env python3
"""ar6_supplementary.py — two supplementary quantities for the autoreviewer response.

(A) Within-topic peak-to-trough: the main text reports the POOLED inverted-U as
    ~5 belief-change points peak-to-trough. The reviewer asks to unbraid pooled
    vs within-topic. This computes the predicted peak-to-trough of the fitted
    quadratic (covariates at means) for the pooled model and for the model with
    20 TF-IDF/KMeans topic fixed effects, over the central IC range [P5,P95].

(B) Continuous IC^2 x density interaction on the |Delta| (movement-magnitude) DV,
    a potentially better-powered outcome than signed change, with the same
    progressive controls as note21b. Reports two-sided and one-sided p (the
    interaction sign is a-priori negative: denser AI argument -> stronger curve).

Headline z(IC)^2 scale; Qwen3.5-ORPO-400 pre-treatment IC. Output: prints +
ar6_supplementary_output.txt.
"""
import io, json, sys, warnings
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"
RAW = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
COV = ("Pre_Belief_Specific", "OpenendedResponseWordCount")


def zs(x):
    x = np.asarray(x, float); sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def load():
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    an = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)
    dem = pd.read_csv(DATA / "costello_demand_composite.csv")
    gpt = pd.read_csv(DATA / "costello_gpt_ic_qwenorpo400.csv")
    gpt["gpt_mean"] = gpt[["ic_r1_logit", "ic_r2_logit", "ic_r3_logit"]].mean(axis=1)
    gpt = gpt.groupby("participantId", as_index=False)["gpt_mean"].mean()
    base = pd.DataFrame({"participantId": [m["participantId"] for m in meta],
                         "IC": q["ic_qwenorpo400_logit"].astype(float).values})
    df = (base.merge(an[["participantId", "DV_BeliefChange_Specific", *COV]], on="participantId")
              .merge(dem[["participantId", "demand_composite"]], on="participantId")
              .merge(gpt, on="participantId"))
    return df.dropna(subset=["IC", "DV_BeliefChange_Specific", *COV,
                             "demand_composite", "gpt_mean"]).reset_index(drop=True)


def topic_labels(df):
    raw = pd.read_csv(RAW, low_memory=False).drop_duplicates("participantId")[
        ["participantId", "conspiracyTheory"]]
    m = df.merge(raw, on="participantId", how="left")
    X = TfidfVectorizer(max_features=500, stop_words="english",
                        ngram_range=(1, 2), min_df=3).fit_transform(
        m["conspiracyTheory"].fillna("").astype(str))
    return KMeans(n_clusters=20, random_state=42, n_init=10).fit(X).labels_


def peak_to_trough(df, tdum=None):
    """Predicted DV peak-to-trough over central IC range [P5,P95], covariates at mean."""
    y = df["DV_BeliefChange_Specific"].to_numpy(float)
    raw = df["IC"].to_numpy(float)
    ic = zs(raw); ic2 = ic ** 2
    cols = [ic, ic2, zs(df[COV[0]].values), zs(df[COV[1]].values)]
    if tdum is not None:
        cols.append(tdum)
    X = sm.add_constant(np.column_stack(cols))
    m = sm.OLS(y, X).fit()
    b0, b1, b2 = m.params[0], m.params[1], m.params[2]
    mu, sd = raw.mean(), raw.std(ddof=0)
    lo, hi = np.percentile(raw, 5), np.percentile(raw, 95)
    grid = np.linspace(lo, hi, 400)
    z = (grid - mu) / sd
    # topic FE contribution is constant across IC (covariate at reference), so
    # peak-to-trough depends only on b1,b2 and the IC grid.
    pred = b0 + b1 * z + b2 * z ** 2
    apex_raw = mu + (-b1 / (2 * b2)) * sd
    return dict(b_ic2=b2, ptt=float(pred.max() - pred.min()),
                apex=float(apex_raw), n=len(df))


def density_interaction_absdv(df, tdum, gm_control=False, fe=False):
    y = np.abs(df["DV_BeliefChange_Specific"].to_numpy(float))
    ic = zs(df["IC"].values); ic2 = ic ** 2
    dens = zs(df["demand_composite"].values)
    cols = [ic, ic2, dens, ic * dens, ic2 * dens,
            zs(df[COV[0]].values), zs(df[COV[1]].values)]
    if gm_control:
        cols.append(zs(df["gpt_mean"].values))
    if fe and tdum is not None:
        cols.append(tdum)
    flat = [c.reshape(-1, 1) if c.ndim == 1 else c for c in cols]
    X = sm.add_constant(np.column_stack(flat))
    m = sm.OLS(y, X).fit()
    b, p2 = m.params[5], m.pvalues[5]            # IC^2 x dens is regressor index 5
    p1 = p2 / 2 if b < 0 else 1 - p2 / 2          # one-sided, predicted b<0
    return b, p2, p1


def run():
    df = load()
    tlab = topic_labels(df)
    tdum = pd.get_dummies(pd.Series(tlab), drop_first=True).astype(float).values

    print("=" * 70)
    print("(A) Peak-to-trough: pooled vs within-topic (central IC range P5-P95)")
    print("=" * 70)
    pool = peak_to_trough(df)
    wtop = peak_to_trough(df, tdum)
    print(f"  Pooled       : beta_IC^2={pool['b_ic2']:+.3f}  apex={pool['apex']:.2f}  "
          f"peak-to-trough = {pool['ptt']:.2f} belief-change points")
    print(f"  Within-topic : beta_IC^2={wtop['b_ic2']:+.3f}  apex={wtop['apex']:.2f}  "
          f"peak-to-trough = {wtop['ptt']:.2f} belief-change points")
    print(f"  ratio within/pooled = {wtop['ptt']/pool['ptt']:.2f}")

    print("\n" + "=" * 70)
    print("(B) Continuous IC^2 x density interaction on |Delta| (magnitude DV)")
    print("=" * 70)
    for lab, gm, fe in [("base", False, False), ("+ AI-side IC", True, False),
                        ("+ topic FE", False, True), ("+ topic FE + AI-IC", True, True)]:
        b, p2, p1 = density_interaction_absdv(df, tdum, gm, fe)
        print(f"  {lab:<22} b(IC^2 x dens)={b:+.3f}  p(2-sided)={p2:.3f}  p(1-sided)={p1:.3f}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    t = buf.getvalue(); sys.stdout.write(t)
    (HERE / "ar6_supplementary_output.txt").write_text(t)
