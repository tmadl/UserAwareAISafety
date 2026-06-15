#!/usr/bin/env python3
"""note21b_density_mechanism.py — SI Note 21, mechanism-tightening analyses.

Autoreviewer theme C / detail #36: the within-Costello evidence-density
split (Note 21) derives from observed AI-side output rather than an
experimental manipulation, and the density-gating reading could in
principle reflect topic difficulty or AI-side argument complexity. The
reviewer asks for "density splits within topics and controls for AI-side
complexity" to align the language with the observational data.

Two complementary tests, both on the headline pre-treatment Qwen3.5-ORPO-400
user-IC and the headline z(IC)^2 curvature scale:

  PART A — continuous interaction. Fit
     DV ~ z(IC) + z(IC)^2 + z(dens) + z(IC).z(dens) + z(IC)^2.z(dens) + covs
  and track the IC^2 x density term (the density-gating-of-curvature
  coefficient; negative = denser AI argument => stronger inverted-U) under
  four specifications: base, + topic FE (k=20 TF-IDF/KMeans clusters),
  + AI-side IC (gpt_mean) main effect, + both.

  PART B — median-split within topic FE. Refit the paper-spec quadratic in
  the low- and high-density halves, adding (i) the 20 topic-cluster dummies
  and (ii) AI-side IC (gpt_mean), to check that the low-flat / high-curved
  pattern is not a topic-sorting or AI-complexity artefact.

Density = AI-side evidence-demand composite (costello_demand_composite.csv).
Topic clusters need the raw Costello CSV for conspiracyTheory text; that
section skips gracefully if the raw file is absent.

Output: prints; writes note21b_density_mechanism_output.txt next to this file.
"""
import io
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "costello2024"
DATA_Q = ROOT / "data" / "ic_qwen3orpo400"
RAW = DATA / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
COVARS = ("Pre_Belief_Specific", "OpenendedResponseWordCount")


def zs(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def load_frame():
    meta = [json.loads(l) for l in open(DATA / "texts_for_scoring.jsonl")]
    ic = pd.read_csv(DATA_Q / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ad = pd.read_csv(DATA / "analysis_data.csv", low_memory=False)
    demand = pd.read_csv(DATA / "costello_demand_composite.csv")
    gpt = pd.read_csv(DATA / "costello_gpt_ic_qwenorpo400.csv")
    gpt["gpt_mean"] = gpt[["ic_r1_logit", "ic_r2_logit", "ic_r3_logit"]].mean(axis=1)
    # The AI-side file has multiple conversation rows per participantId; collapse
    # to one mean AI-side IC per participant so the baseline split matches Note 21.
    gpt = gpt.groupby("participantId", as_index=False)["gpt_mean"].mean()

    base = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "IC":             ic["ic_qwenorpo400_logit"].astype(float).values})
    df = (base.merge(ad[["participantId", "DV_BeliefChange_Specific", *COVARS]],
                     on="participantId", how="inner")
              .merge(demand[["participantId", "demand_composite"]],
                     on="participantId", how="inner")
              .merge(gpt[["participantId", "gpt_mean"]],
                     on="participantId", how="inner"))
    df = df.dropna(subset=["IC", "DV_BeliefChange_Specific", *COVARS,
                           "demand_composite", "gpt_mean"]).reset_index(drop=True)
    return df


def topic_dummies(df):
    """20 TF-IDF/KMeans clusters from raw conspiracyTheory text, aligned to df.
    Returns (dummies ndarray or None, n_clusters)."""
    if not RAW.exists():
        return None, 0
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    raw = pd.read_csv(RAW, low_memory=False).drop_duplicates(
        "participantId", keep="first")[["participantId", "conspiracyTheory"]]
    m = df.merge(raw, on="participantId", how="left")
    txt = m["conspiracyTheory"].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=500, stop_words="english",
                            ngram_range=(1, 2), min_df=3)
    X = tfidf.fit_transform(txt)
    km = KMeans(n_clusters=20, random_state=42, n_init=10).fit(X)
    return km.labels_, 20


def fit_quad(y, ic_raw, pre, wc, extra=None):
    """Paper-spec quadratic on z(IC)^2 scale; returns b_IC2, p, BF, apex."""
    ic = zs(ic_raw); ic2 = ic ** 2
    cols = [ic, ic2, zs(pre), zs(wc)]
    if extra is not None:
        cols += list(extra)
    X_q = sm.add_constant(np.column_stack(cols))
    X_l = sm.add_constant(np.column_stack([c for i, c in enumerate(cols) if i != 1]))
    m_q, m_l = sm.OLS(y, X_q).fit(), sm.OLS(y, X_l).fit()
    bf = float(np.exp((m_l.bic - m_q.bic) / 2))
    b1, b2 = m_q.params[1], m_q.params[2]
    sd = np.std(ic_raw, ddof=0)
    apex = np.mean(ic_raw) + (-b1 / (2 * b2)) * sd if abs(b2) > 1e-12 else np.nan
    return dict(b_ic2=b2, p_ic2=m_q.pvalues[2], BF10=bf, apex=apex, n=len(y))


def part_a(df, tdum):
    print("\n" + "=" * 72)
    print("  PART A — continuous IC^2 x density interaction")
    print("=" * 72)
    y = df["DV_BeliefChange_Specific"].to_numpy(float)
    ic = zs(df["IC"].values); ic2 = ic ** 2
    dens = zs(df["demand_composite"].values)
    gm = zs(df["gpt_mean"].values)
    pre = zs(df[COVARS[0]].values); wc = zs(df[COVARS[1]].values)
    icXd, ic2Xd = ic * dens, ic2 * dens

    print(f"  r(density composite, AI-side IC gpt_mean) = "
          f"{np.corrcoef(dens, gm)[0, 1]:+.3f}")
    print(f"\n  {'Specification':<34}{'b(IC^2 x dens)':>15}{'p':>9}{'b(IC^2)':>10}")
    base = [ic, ic2, dens, icXd, ic2Xd, pre, wc]
    specs = [("base", base),
             ("+ topic FE (k=20)", base + ([tdum_block(tdum)] if tdum is not None else [])),
             ("+ AI-side IC (gpt_mean)", base + [gm]),
             ("+ topic FE + AI-side IC",
              base + [gm] + ([tdum_block(tdum)] if tdum is not None else []))]
    for label, cols in specs:
        flat = []
        for c in cols:
            if c.ndim == 1:
                flat.append(c.reshape(-1, 1))
            else:
                flat.append(c)
        X = sm.add_constant(np.column_stack(flat))
        m = sm.OLS(y, X).fit()
        # IC^2 x dens is the 5th regressor (index 5: const,ic,ic2,dens,icXd,ic2Xd)
        b_int, p_int = m.params[5], m.pvalues[5]
        b_ic2 = m.params[2]
        note = "" if (tdum is not None or "topic" not in label) else "  [skipped: no raw text]"
        print(f"  {label:<34}{b_int:>+15.3f}{p_int:>9.3f}{b_ic2:>+10.3f}{note}")


def tdum_block(tdum):
    return pd.get_dummies(pd.Series(tdum), drop_first=True).astype(float).values


def part_b(df, tdum):
    print("\n" + "=" * 72)
    print("  PART B — median-density split, within topic FE + AI-side IC")
    print("=" * 72)
    med = df["demand_composite"].median()
    df = df.assign(_half=np.where(df["demand_composite"] <= med, "low", "high"))
    has_fe = tdum is not None
    if has_fe:
        df = df.assign(_topic=tdum)

    for control, label in [("none", "paper spec (no FE, no AI-IC)"),
                           ("fe", "+ topic FE (k=20)"),
                           ("ai", "+ AI-side IC (gpt_mean)"),
                           ("both", "+ topic FE + AI-side IC")]:
        if control in ("fe", "both") and not has_fe:
            print(f"\n  [{label}: skipped — needs raw Costello text]")
            continue
        print(f"\n  {label}:")
        print(f"    {'half':<6}{'n':>5}{'b_IC^2':>9}{'p':>8}{'BF10':>10}{'apex':>7}")
        for half in ("low", "high"):
            s = df[df["_half"] == half]
            y = s["DV_BeliefChange_Specific"].to_numpy(float)
            extra = []
            if control in ("ai", "both"):
                extra.append(zs(s["gpt_mean"].values))
            if control in ("fe", "both"):
                # dummies restricted to clusters present in this half
                d = pd.get_dummies(s["_topic"], drop_first=True).astype(float).values
                extra.append(d)
            extra_cols = []
            for e in extra:
                extra_cols.append(e.reshape(-1, 1) if e.ndim == 1 else e)
            r = fit_quad(y, s["IC"].values, s[COVARS[0]].values,
                         s[COVARS[1]].values,
                         extra=extra_cols if extra_cols else None)
            print(f"    {half:<6}{r['n']:>5}{r['b_ic2']:>+9.2f}{fmt_p(r['p_ic2']):>8}"
                  f"{r['BF10']:>10.2f}{r['apex']:>+7.2f}")


def run():
    df = load_frame()
    print(f"SI Note 21b — density-mechanism tightening")
    print(f"Complete-cases N = {len(df)}")
    tdum, k = topic_dummies(df)
    print(f"Topic clusters: {'k=' + str(k) if tdum is not None else 'UNAVAILABLE (no raw text)'}")
    part_a(df, tdum)
    part_b(df, tdum)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    t = buf.getvalue()
    sys.stdout.write(t)
    (HERE / "note21b_density_mechanism_output.txt").write_text(t)
    sys.stdout.write(f"\nWrote {HERE / 'note21b_density_mechanism_output.txt'}\n")
