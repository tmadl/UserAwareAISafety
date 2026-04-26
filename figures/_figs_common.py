"""Shared loading + style helpers for figure-generation scripts.

Lightweight Costello-only loader; the public release figures only require
Costello data, so we don't bring in cheng/salvi/boissin/tessler loaders.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_Q4 = DATA / "ic_qwen3orpo400"
FIG_OUT = ROOT / "figures"

PNAS_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
}

BLUE = "#2166AC"
RED = "#B2182B"
GRAY = "#969696"
LIGHTBLUE = "#92C5DE"

# PNAS column widths (inches): single = 8.7cm ~ 3.42", double = 17.8cm ~ 7.01"
COL1 = 3.42
COL2 = 7.01


def load_costello_q400():
    """Costello: analysis_data + Q400 logit-EV IC, aligned via texts_for_scoring.jsonl."""
    meta_path = DATA / "costello2024" / "texts_for_scoring.jsonl"
    q_path = DATA_Q4 / "costello_texts_for_scoring_initial_qwenorpo400.csv"
    ana_path = DATA / "costello2024" / "analysis_data.csv"

    meta = [json.loads(l) for l in open(meta_path)]
    pids = [m["participantId"] for m in meta]
    q = pd.read_csv(q_path)
    assert len(q) == len(meta), f"Q400 len mismatch: {len(q)} vs {len(meta)}"
    q_df = pd.DataFrame({
        "participantId": pids,
        "IC_q4": pd.to_numeric(q["ic_qwenorpo400_logit"], errors="coerce").values,
    })
    ana = pd.read_csv(ana_path)
    df = ana.merge(q_df, on="participantId", how="inner").dropna(
        subset=["IC_q4", "DV_BeliefChange_Specific"]
    )
    return df


def bootstrap_quadratic_ci(x, y, n_boot=2000, seed=0, conf=0.95):
    """Bootstrap quadratic fit, return (xgrid, yhat, lo, hi)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xgrid = np.linspace(x.min(), x.max(), 100)
    preds = np.zeros((n_boot, len(xgrid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        X = np.column_stack([np.ones_like(xb), xb, xb ** 2])
        beta, *_ = np.linalg.lstsq(X, yb, rcond=None)
        preds[b] = beta[0] + beta[1] * xgrid + beta[2] * xgrid ** 2
    lo = np.percentile(preds, (1 - conf) / 2 * 100, axis=0)
    hi = np.percentile(preds, (1 + conf) / 2 * 100, axis=0)
    yhat = preds.mean(axis=0)
    return xgrid, yhat, lo, hi


def rank_quintiles(df, ic_col="IC_q4"):
    df = df.copy()
    df["_rank"] = df[ic_col].rank(method="first")
    df["q"] = pd.qcut(df["_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return df
