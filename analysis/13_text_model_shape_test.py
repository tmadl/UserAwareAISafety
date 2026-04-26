#!/usr/bin/env python3
"""13_text_model_shape_test.py — Generic text-model shape test for SI Note 26.

Reviewer attack this addresses:
  "IC is a relabelling of whatever semantic signal predicts belief change.
   A plain TF-IDF or sentence-embedding model, fit to predict DeltaBelief
   from the same pre-treatment text, would recover the same moderation."

This script runs the two diagnostics reported in Note 26:

  (A) Shape-specificity scaling. Four text representations of ascending
      capacity (TF-IDF 2k, MiniLM 384, OpenAI te3-small 1536, te3-large 3072)
      each ridge-fit to DeltaBelief via 5-fold CV; the resulting out-of-sample
      predictions are the shape-test input. More capacity yields better linear
      prediction of DeltaBelief, but not a stronger inverted-U.

  (B) Predicted-IC diagnostic. The SAME te3-large features, ridge-steered at
      IC instead of at DeltaBelief, reproduce the inverted-U. Localises the
      curvature to the IC direction in embedding space, not to the direction
      that maximally predicts the outcome.

Produces the two tables in SI Note 26 (tab:text_model_shape, tab:predicted_ic).

Cached OpenAI embeddings live in `data/openai_embeddings/`. If missing, the
script re-fetches from the OpenAI API (requires OPENAI_API_KEY).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


class _WandbStub:
    def __getattr__(self, name):
        return _WandbStub()

    def __call__(self, *a, **kw):
        return _WandbStub()


_stub = _WandbStub()
sys.modules.setdefault("wandb", _stub)
sys.modules.setdefault("wandb.sdk", _stub)
sys.modules.setdefault("wandb.sdk.lib", _stub)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
AD_CSV = ROOT / "data/costello2024/analysis_data.csv"
IC_CSV = ROOT / "data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv"
META_JSONL = ROOT / "data/costello2024/texts_for_scoring.jsonl"
OAI_DIR = ROOT / "data/openai_embeddings"

OPENAI_MODELS = {
    "te3-small": ("text-embedding-3-small", OAI_DIR / "costello_te3_small.npy"),
    "te3-large": ("text-embedding-3-large", OAI_DIR / "costello_te3_large.npy"),
}
OPENAI_BATCH = 64


def zs(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def load_df():
    meta = [json.loads(l) for l in open(META_JSONL)]
    ic = pd.read_csv(IC_CSV)
    base = pd.DataFrame({
        "participantId": [m["participantId"] for m in meta],
        "text": [m.get("text_initial", "") for m in meta],
        "IC": ic["ic_qwenorpo400_logit"].astype(float).values,
    })
    ad = pd.read_csv(AD_CSV, low_memory=False)
    df = base.merge(
        ad[["participantId", "DV_BeliefChange_Specific",
            "Pre_Belief_Specific", "OpenendedResponseWordCount"]],
        on="participantId", how="inner",
    )
    df = df.drop_duplicates("participantId").reset_index(drop=True)
    df = df.dropna(subset=["IC", "DV_BeliefChange_Specific",
                           "Pre_Belief_Specific",
                           "OpenendedResponseWordCount"]).reset_index(drop=True)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def kfold_ridge_predict(X, y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yhat = np.full(len(y), np.nan, dtype=float)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], y[tr])
        yhat[te] = m.predict(X[te])
    return yhat


def tfidf_features(texts, max_features=2000):
    vec = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2),
        min_df=3, sublinear_tf=True, lowercase=True, strip_accents="unicode",
    )
    return vec.fit_transform(texts).toarray()


def minilm_embeddings(texts, batch=32, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(pooled.cpu().numpy())
    return np.vstack(out)


def openai_embeddings(texts, cache_path: Path, model_name: str):
    """Load cached .npy if present, else re-fetch from OpenAI API.

    The public repo ships the cached embeddings under data/openai_embeddings/;
    re-fetching is only needed for reproducibility from scratch and requires
    OPENAI_API_KEY in the environment.
    """
    if cache_path.exists():
        E = np.load(cache_path)
        if E.shape[0] == len(texts):
            print(f"  [cache] {cache_path.name} shape={E.shape}")
            return E
        print(f"  [cache] shape mismatch ({E.shape[0]} != {len(texts)}), re-fetching")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print(f"ERROR: cached {cache_path.name} missing and OPENAI_API_KEY not set")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=key)
    out = []
    n = len(texts)
    t0 = time.time()
    for i in range(0, n, OPENAI_BATCH):
        chunk = [c if c.strip() else " " for c in texts[i : i + OPENAI_BATCH]]
        for attempt in range(4):
            try:
                resp = client.embeddings.create(model=model_name, input=chunk)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"    [batch {i}] retry in {wait}s ({e})")
                time.sleep(wait)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        out.append(vecs)
        if i % (OPENAI_BATCH * 4) == 0:
            elapsed = time.time() - t0
            rate = (i + len(chunk)) / max(elapsed, 1e-3)
            eta = (n - (i + len(chunk))) / max(rate, 1e-3)
            print(f"    [{i + len(chunk)}/{n}]  rate={rate:.1f}/s  eta={eta:.0f}s")
    E = np.vstack(out)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, E)
    print(f"  [cache] wrote {cache_path.name} shape={E.shape}")
    return E


def fit_quadratic_on_x(df, x_col, controls=("Pre_Belief_Specific", "OpenendedResponseWordCount")):
    """DV ~ x + x^2 + controls, all z-scored. Returns betas, p-values, BF10."""
    s = df.dropna(subset=["DV_BeliefChange_Specific", x_col, *controls]).copy()
    y = s["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    raw = s[x_col].to_numpy(dtype=float)
    x_z = zs(raw)
    x2_z = zs(raw ** 2)
    cov_z = [zs(s[c].to_numpy(dtype=float)) for c in controls]
    X1 = sm.add_constant(np.column_stack([x_z, *cov_z]))
    X2 = sm.add_constant(np.column_stack([x_z, x2_z, *cov_z]))
    m1 = sm.OLS(y, X1).fit()
    m2 = sm.OLS(y, X2).fit()
    bf = float(np.exp((m1.bic - m2.bic) / 2))
    return dict(
        n=len(s),
        beta_lin=float(m2.params[1]), p_lin=float(m2.pvalues[1]),
        beta_q=float(m2.params[2]),   p_q=float(m2.pvalues[2]),
        BF10=bf, r2_quad=float(m2.rsquared), r2_lin=float(m1.rsquared),
    )


def held_out_stats(y, yhat):
    mask = ~np.isnan(yhat) & ~np.isnan(y)
    y, yhat = y[mask], yhat[mask]
    r = float(np.corrcoef(y, yhat)[0, 1])
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return r, r2


def main():
    df = load_df()
    print(f"N = {len(df)}")
    y = df["DV_BeliefChange_Specific"].to_numpy(dtype=float)
    ic = df["IC"].to_numpy(dtype=float)

    # ---- IC canonical (reference row) ----
    r_ic = fit_quadratic_on_x(df, "IC")
    print(f"\n[IC canonical] beta_quad = {r_ic['beta_q']:+.2f}  p = {r_ic['p_q']:.3g}  "
          f"BF10 = {r_ic['BF10']:.3g}  R2(quad) = {r_ic['r2_quad']:.4f}")

    # =====================================================================
    # Pillar (A): Shape-specificity scaling across four text representations
    # =====================================================================
    print("\n" + "=" * 80)
    print("  (A) Shape-specificity scaling")
    print("=" * 80)

    # TF-IDF
    print("\n-- TF-IDF (2,000 word uni-/bi-grams) --")
    X_tfidf = tfidf_features(df["text"].tolist(), max_features=2000)
    yhat_tfidf = kfold_ridge_predict(X_tfidf, y)
    r, r2 = held_out_stats(y, yhat_tfidf)
    df["yhat_tfidf"] = yhat_tfidf
    s_tfidf = fit_quadratic_on_x(df, "yhat_tfidf")
    print(f"  OOS r = {r:+.3f}  R2 = {r2:+.4f}  |  "
          f"beta_quad = {s_tfidf['beta_q']:+.2f}  p = {s_tfidf['p_q']:.3f}  "
          f"BF10 = {s_tfidf['BF10']:.3g}")

    # MiniLM
    print("\n-- MiniLM (384-dim, sentence-transformers/all-MiniLM-L6-v2) --")
    X_emb = minilm_embeddings(df["text"].tolist())
    yhat_emb = kfold_ridge_predict(X_emb, y)
    r, r2 = held_out_stats(y, yhat_emb)
    df["yhat_minilm"] = yhat_emb
    s_minilm = fit_quadratic_on_x(df, "yhat_minilm")
    print(f"  OOS r = {r:+.3f}  R2 = {r2:+.4f}  |  "
          f"beta_quad = {s_minilm['beta_q']:+.2f}  p = {s_minilm['p_q']:.3f}  "
          f"BF10 = {s_minilm['BF10']:.3g}")

    # OpenAI te3-small
    print("\n-- OpenAI text-embedding-3-small (1,536-dim) --")
    E_small = openai_embeddings(df["text"].tolist(),
                                OPENAI_MODELS["te3-small"][1],
                                OPENAI_MODELS["te3-small"][0])
    yhat_small = kfold_ridge_predict(E_small, y)
    r, r2 = held_out_stats(y, yhat_small)
    df["yhat_te3small"] = yhat_small
    s_small = fit_quadratic_on_x(df, "yhat_te3small")
    print(f"  OOS r = {r:+.3f}  R2 = {r2:+.4f}  |  "
          f"beta_quad = {s_small['beta_q']:+.2f}  p = {s_small['p_q']:.3f}  "
          f"BF10 = {s_small['BF10']:.3g}")

    # OpenAI te3-large
    print("\n-- OpenAI text-embedding-3-large (3,072-dim) --")
    E_large = openai_embeddings(df["text"].tolist(),
                                OPENAI_MODELS["te3-large"][1],
                                OPENAI_MODELS["te3-large"][0])
    yhat_large = kfold_ridge_predict(E_large, y)
    r_l_dv, r2_l_dv = held_out_stats(y, yhat_large)
    df["yhat_te3large"] = yhat_large
    s_large = fit_quadratic_on_x(df, "yhat_te3large")
    print(f"  OOS r = {r_l_dv:+.3f}  R2 = {r2_l_dv:+.4f}  |  "
          f"beta_quad = {s_large['beta_q']:+.2f}  p = {s_large['p_q']:.3f}  "
          f"BF10 = {s_large['BF10']:.3g}")

    # Scaling table (tab:text_model_shape)
    print("\n---- Scaling table (tab:text_model_shape) ----")
    header = f"  {'Model':<22}{'Dim':>8}{'OOS r':>10}{'OOS R2':>10}{'beta_q':>10}{'p':>10}{'BF10':>10}"
    print(header)
    print(f"  {'IC (Q400)':<22}{'--':>8}{'--':>10}{'--':>10}"
          f"{r_ic['beta_q']:>10.2f}{r_ic['p_q']:>10.3g}{r_ic['BF10']:>10.3g}")
    rows = [
        ("TF-IDF",    2000, "yhat_tfidf",    s_tfidf),
        ("MiniLM",    384,  "yhat_minilm",   s_minilm),
        ("te3-small", 1536, "yhat_te3small", s_small),
        ("te3-large", 3072, "yhat_te3large", s_large),
    ]
    for name, dim, col, st in rows:
        yh = df[col].to_numpy(dtype=float)
        rr, rr2 = held_out_stats(y, yh)
        print(f"  {name:<22}{dim:>8}{rr:>+10.3f}{rr2:>+10.4f}"
              f"{st['beta_q']:>10.2f}{st['p_q']:>10.3g}{st['BF10']:>10.3g}")

    # =====================================================================
    # Pillar (B): Predicted-IC diagnostic with te3-large
    # =====================================================================
    print("\n" + "=" * 80)
    print("  (B) Predicted-IC diagnostic: te3-large -> IC vs te3-large -> DV")
    print("=" * 80)

    yhat_ic_from_emb = kfold_ridge_predict(E_large, ic)
    r_ic_pred, _ = held_out_stats(ic, yhat_ic_from_emb)
    df["yhat_ic_from_emb"] = yhat_ic_from_emb
    s_ic_from_emb = fit_quadratic_on_x(df, "yhat_ic_from_emb")
    s_dv_from_emb = s_large  # already computed: te3-large ridge-steered at DV

    print(f"\n  Ridge(te3-large -> IC): OOS r vs IC  = {r_ic_pred:+.3f}")
    print(f"  Ridge(te3-large -> DV): OOS r vs DV  = {r_l_dv:+.3f}")
    print(f"\n  Shape test on DV:")
    print(f"    predicted-IC input:  beta_q = {s_ic_from_emb['beta_q']:+.2f}  "
          f"p = {s_ic_from_emb['p_q']:.3g}  BF10 = {s_ic_from_emb['BF10']:.3g}")
    print(f"    predicted-DV input:  beta_q = {s_dv_from_emb['beta_q']:+.2f}  "
          f"p = {s_dv_from_emb['p_q']:.3g}  BF10 = {s_dv_from_emb['BF10']:.3g}")

    print("\n---- Predicted-IC table (tab:predicted_ic) ----")
    print(f"  {'Ridge target (te3-large -> y)':<36}{'OOS r vs y':>12}"
          f"{'beta_q(DV)':>12}{'p':>10}{'BF10':>10}")
    print(f"  {'IC (theory-driven construct)':<36}{r_ic_pred:>+12.3f}"
          f"{s_ic_from_emb['beta_q']:>+12.2f}{s_ic_from_emb['p_q']:>10.3g}"
          f"{s_ic_from_emb['BF10']:>10.3g}")
    print(f"  {'DeltaBelief (outcome)':<36}{r_l_dv:>+12.3f}"
          f"{s_dv_from_emb['beta_q']:>+12.2f}{s_dv_from_emb['p_q']:>10.3g}"
          f"{s_dv_from_emb['BF10']:>10.3g}")


if __name__ == "__main__":
    main()
