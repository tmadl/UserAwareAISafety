#!/usr/bin/env python3
"""fig2_costello_deepdive.py — main-text Figure 2 (Costello deep-dive).

Two-panel figure:
  (A) Large-change and adverse-movement rates by IC quintile.
  (B) Drop-bottom-Q enrichment curve with bootstrap 95% CI bands;
      LOSO operating-point numbers and AUC annotated.

Reproduces fig:costello in the main text.

LOSO panel needs the raw Costello CSV for the StudyNumber column
(`AllDataForPublication.PPI.8.28.24.csv`). If raw is absent, panel B
falls back to in-sample-only annotations (LOSO numbers omitted).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figs_common import (
    BLUE, COL2, DATA, FIG_OUT, GRAY, PNAS_RC, RED,
    load_costello_q400, rank_quintiles,
)

plt.rcParams.update(PNAS_RC)


def ensure_studynumber(df):
    """StudyNumber is in analysis_data.csv; this is a defensive no-op fallback
    that pulls it from the raw Costello CSV if (somehow) it was dropped upstream."""
    if "StudyNumber" in df.columns and df["StudyNumber"].notna().any():
        return df
    raw = DATA / "costello2024" / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
    if not raw.exists():
        return df
    raw_df = pd.read_csv(raw, low_memory=False).drop_duplicates("participantId", keep="first")
    return df.merge(raw_df[["participantId", "StudyNumber"]], on="participantId", how="left")


def panel_A_quintile_bars(ax, df):
    df_q = rank_quintiles(df, "IC_q4")
    df_q["large_change"] = (df_q["DV_BeliefChange_Specific"] >= 20).astype(int)
    df_q["adverse_move"] = (df_q["DV_BeliefChange_Specific"] <= -5).astype(int)
    g = df_q.groupby("q").agg(
        lc_rate=("large_change", "mean"),
        am_rate=("adverse_move", "mean"),
        n=("q", "size"),
    ).reset_index()
    qs = g["q"].values
    x = np.arange(len(qs))
    width = 0.38
    ax.bar(x - width / 2, g["lc_rate"] * 100, width, color=BLUE,
           label=r"Large belief change ($\geq$20 pts)", edgecolor="black", linewidth=0.3)
    ax.bar(x + width / 2, g["am_rate"] * 100, width, color=RED,
           label=r"Adverse movement ($\leq-5$ pts)", edgecolor="black", linewidth=0.3)
    for i in range(len(qs)):
        ratio = g["lc_rate"].iloc[i] / max(g["am_rate"].iloc[i], 1e-6)
        ymax = max(g["lc_rate"].iloc[i], g["am_rate"].iloc[i]) * 100
        ax.text(i, ymax + 1.5, f"{ratio:.1f}:1", ha="center", va="bottom",
                fontsize=5.5, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{q}\n($n$={g['n'].iloc[i]})" for i, q in enumerate(qs)], fontsize=6)
    ax.set_ylabel("Percentage of participants in quintile")
    ax.set_xlabel(r"Integrative-complexity quintile (low $\rightarrow$ high)")
    ax.set_title("A  Large-change and adverse-movement rates by IC quintile",
                 loc="left", fontweight="bold", fontsize=7)
    ax.legend(loc="upper right", frameon=False, fontsize=6)
    ax.set_ylim(0, max(g["lc_rate"].max(), g["am_rate"].max()) * 100 * 1.25)


def _insample_enrichment(df):
    df_q = rank_quintiles(df, "IC_q4")
    df_q["large_change"] = (df_q["DV_BeliefChange_Specific"] >= 20).astype(int)
    df_q["adverse_move"] = (df_q["DV_BeliefChange_Specific"] <= -5).astype(int)
    total_lc = df_q["large_change"].sum()
    total_am = df_q["adverse_move"].sum()
    N = len(df_q)
    rows = []
    for t in [0, 1, 2, 3, 4]:
        kept = df_q[df_q["q"] > t]
        dropped = df_q[df_q["q"] <= t]
        rows.append({
            "drop_q_up_to": t,
            "frac_kept": len(kept) / N,
            "lc_kept_frac": kept["large_change"].sum() / total_lc,
            "am_excluded_frac": dropped["adverse_move"].sum() / total_am,
        })
    return pd.DataFrame(rows)


def _loso_enrichment(df, study_col="StudyNumber"):
    sub = df.dropna(subset=[study_col]).copy()
    studies = sorted(sub[study_col].unique())
    pool = []
    for s in studies:
        tr = sub[sub[study_col] != s]
        te = sub[sub[study_col] == s].copy().reset_index(drop=True)
        cuts = [tr["IC_q4"].quantile(q / 5) for q in (1, 2, 3, 4)]
        q_te = np.ones(len(te), dtype=int)
        for c in cuts:
            q_te += (te["IC_q4"].values > c).astype(int)
        te["q_loso"] = q_te
        te["large_change"] = (te["DV_BeliefChange_Specific"] >= 20).astype(int)
        te["adverse_move"] = (te["DV_BeliefChange_Specific"] <= -5).astype(int)
        pool.append(te)
    pool = pd.concat(pool, ignore_index=True)
    total_lc = pool["large_change"].sum()
    total_am = pool["adverse_move"].sum()
    N = len(pool)
    rows = []
    for t in [0, 1, 2, 3, 4]:
        kept = pool[pool["q_loso"] > t]
        dropped = pool[pool["q_loso"] <= t]
        rows.append({
            "drop_q_up_to": t,
            "frac_kept": len(kept) / N,
            "lc_kept_frac": kept["large_change"].sum() / total_lc,
            "am_excluded_frac": dropped["adverse_move"].sum() / total_am,
        })
    return pd.DataFrame(rows)


def _bootstrap_enrichment(df, n_boot=2000, seed=20260419):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["large_change"] = (df["DV_BeliefChange_Specific"] >= 20).astype(int)
    df["adverse_move"] = (df["DV_BeliefChange_Specific"] <= -5).astype(int)
    lc_boot = np.zeros((n_boot, 5))
    am_boot = np.zeros((n_boot, 5))
    fk_boot = np.zeros((n_boot, 5))
    N = len(df)
    idx_all = np.arange(N)
    for b in range(n_boot):
        resamp = df.iloc[rng.choice(idx_all, N, replace=True)].copy()
        rq = rank_quintiles(resamp, "IC_q4")
        total_lc = rq["large_change"].sum()
        total_am = rq["adverse_move"].sum()
        for t in range(5):
            kept = rq[rq["q"] > t]
            dropped = rq[rq["q"] <= t]
            fk_boot[b, t] = len(kept) / N
            lc_boot[b, t] = kept["large_change"].sum() / max(total_lc, 1)
            am_boot[b, t] = dropped["adverse_move"].sum() / max(total_am, 1)
    return {
        "fk_mean": fk_boot.mean(axis=0),
        "lc_lo": np.percentile(lc_boot, 2.5, axis=0),
        "lc_hi": np.percentile(lc_boot, 97.5, axis=0),
        "am_lo": np.percentile(am_boot, 2.5, axis=0),
        "am_hi": np.percentile(am_boot, 97.5, axis=0),
    }


def panel_B_enrichment(ax, df):
    e_in = _insample_enrichment(df)
    has_loso = "StudyNumber" in df.columns and df["StudyNumber"].notna().any()
    e_oos = _loso_enrichment(df, "StudyNumber") if has_loso else None
    ci = _bootstrap_enrichment(df)

    print("\n[Panel B] In-sample enrichment:")
    print(e_in.to_string(index=False))
    if has_loso:
        print("\n[Panel B] LOSO enrichment (pooled):")
        print(e_oos.to_string(index=False))
    else:
        print("\n[Panel B] LOSO skipped (raw Costello CSV missing — see note22_loso_enrichment.py).")

    xv = e_in["frac_kept"].values * 100

    ax.fill_between(xv, ci["lc_lo"] * 100, ci["lc_hi"] * 100,
                    color=BLUE, alpha=0.18, linewidth=0)
    ax.fill_between(xv, ci["am_lo"] * 100, ci["am_hi"] * 100,
                    color=RED, alpha=0.18, linewidth=0)
    ax.plot(xv, e_in["lc_kept_frac"].values * 100, "-o",
            color=BLUE, ms=5, lw=1.4,
            label=r"Large changes preserved ($\geq 20$ pts)")
    ax.plot(xv, e_in["am_excluded_frac"].values * 100, "-s",
            color=RED, ms=5, lw=1.4,
            label=r"Adverse cases captured ($\leq -5$ pts)")

    idx = 1
    lc_in = e_in["lc_kept_frac"].iloc[idx] * 100
    am_in = e_in["am_excluded_frac"].iloc[idx] * 100
    ax.axvline(xv[idx], color=GRAY, lw=0.5, ls=":")

    if has_loso:
        lc_oos = e_oos["lc_kept_frac"].iloc[idx] * 100
        am_oos = e_oos["am_excluded_frac"].iloc[idx] * 100
        ann = (
            f"Flag lowest IC quintile (20% flagged for safeguards)\n"
            f"  Adverse cases captured: {am_in:.1f}% (LOSO {am_oos:.1f}%)\n"
            f"  Large changes preserved: {lc_in:.1f}% (LOSO {lc_oos:.1f}%)\n"
            r"  AUC (IC $\to$ adverse, LOSO) = 0.69"
        )
    else:
        ann = (
            f"Flag lowest IC quintile (20% flagged for safeguards)\n"
            f"  Adverse cases captured: {am_in:.1f}%\n"
            f"  Large changes preserved: {lc_in:.1f}%\n"
            r"  LOSO operating point in note22_loso_enrichment.py"
        )
    ax.text(0.5, 0.05, ann, transform=ax.transAxes,
            fontsize=4.8, color="black", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=GRAY, lw=0.4, alpha=0.95))

    ax.set_xlabel("Unflagged cohort share (%)")
    ax.set_ylabel("Outcome (%)")
    ax.set_title("B  Illustrative enrichment: flag by IC quintile",
                 loc="left", fontweight="bold", fontsize=7)
    ax.set_xlim(100, 15)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=False, fontsize=5.5)


def main():
    df = load_costello_q400()
    df = ensure_studynumber(df)
    print(f"Costello N = {len(df)}")
    if "StudyNumber" in df.columns:
        vc = df["StudyNumber"].value_counts().sort_index().to_dict()
        print(f"StudyNumber counts: {vc}")

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9))
    panel_A_quintile_bars(axes[0], df)
    panel_B_enrichment(axes[1], df)
    fig.tight_layout(pad=0.4, w_pad=1.5)

    FIG_OUT.mkdir(exist_ok=True)
    fig.savefig(FIG_OUT / "fig2_costello_deepdive.pdf")
    fig.savefig(FIG_OUT / "fig2_costello_deepdive.png")
    print("Wrote fig2_costello_deepdive.{pdf,png}")


if __name__ == "__main__":
    main()
