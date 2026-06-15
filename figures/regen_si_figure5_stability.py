"""Regenerate figures/si_figure5_scoring_stability with panel C on the
paper-canonical z(IC)^2 scale (was: z(IC^2) scale, showing -15.2/-13.2).

Identical to scripts/figures/si_figures_q400.py make_fig5 except:
  - panel C: ic2 regressor is z(IC)**2 (ddof=1), so Q400 bars land at
    -1.97 (signed) and -1.72 (|Delta|), matching the caption
  - bar value labels use 2 decimals
  - y-label says canonical z(IC)^2 scale
  - output -> paper/PNAS_Nexus_v3/figures
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, "/mnt/workvm/UserAwareAISafety/scripts/figures")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

from _common import PNAS_RC, BLUE, RED, GRAY, COL2

plt.rcParams.update(PNAS_RC)
DATA = Path("/mnt/workvm/UserAwareAISafety/data")
OUT = Path("/mnt/workvm/UserAwareAISafety/paper/PNAS_Nexus_v3/figures")


def add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


def _load_checkpoint_ic(ckpt_tag):
    f = DATA / f"ic_qwen3orpo{ckpt_tag}/costello_texts_for_scoring_initial_qwenorpo{ckpt_tag}.csv"
    if not f.exists():
        return None
    meta_file = DATA / "costello2024/texts_for_scoring.jsonl"
    pids = [json.loads(l)["participantId"] for l in open(meta_file)]
    q = pd.read_csv(f)
    assert len(q) == len(pids)
    col = f"ic_qwenorpo{ckpt_tag}_logit"
    if col not in q.columns:
        return None
    return pd.DataFrame({"participantId": pids,
                         "IC": pd.to_numeric(q[col], errors="coerce").values})


def main():
    ana = pd.read_csv(DATA / "costello2024/analysis_data.csv")
    ckpts = [("200", BLUE), ("400", RED)]
    entries = []
    for tag, c in ckpts:
        ic_df = _load_checkpoint_ic(tag)
        if ic_df is None:
            print(f"  checkpoint {tag} missing — skipping")
            continue
        df = ana.merge(ic_df, on="participantId").dropna(subset=["IC", "DV_BeliefChange_Specific"])
        df["absdv"] = df.DV_BeliefChange_Specific.abs()
        entries.append((tag, c, df))
        print(f"  Q{tag}: N = {len(df)}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(COL2, 2.9))

    # Panel A: signed-DV unadjusted quadratic fit overlay on z-scored IC
    for tag, c, df in entries:
        ic = df.IC.values
        icz = (ic - ic.mean()) / ic.std()
        y = df.DV_BeliefChange_Specific.values
        xgrid = np.linspace(icz.min(), icz.max(), 200)
        coef = np.polyfit(icz, y, 2)
        ax1.plot(xgrid, np.polyval(coef, xgrid), color=c, lw=1.6, label=f"Q{tag}")
    ax1.axhline(0, color=GRAY, lw=0.5, ls="--")
    ax1.set_xlabel("Integrative complexity (z-scored)")
    ax1.set_ylabel("Predicted belief change (signed)")
    ax1.legend(frameon=False, fontsize=6, loc="upper right")
    ax1.set_title("Signed-DV quadratic fits", fontsize=7, fontweight="bold")
    add_panel_label(ax1, "A")

    # Panel B: |Delta| unadjusted quadratic fit overlay
    for tag, c, df in entries:
        ic = df.IC.values
        icz = (ic - ic.mean()) / ic.std()
        y = df.absdv.values
        xgrid = np.linspace(icz.min(), icz.max(), 200)
        coef = np.polyfit(icz, y, 2)
        ax2.plot(xgrid, np.polyval(coef, xgrid), color=c, lw=1.6, label=f"Q{tag}")
    ax2.axhline(0, color=GRAY, lw=0.5, ls="--")
    ax2.set_xlabel("Integrative complexity (z-scored)")
    ax2.set_ylabel(r"Predicted $|\Delta\mathrm{belief}|$")
    ax2.legend(frameon=False, fontsize=6, loc="upper right")
    ax2.set_title(r"$|\Delta|$ quadratic fits", fontsize=7, fontweight="bold")
    add_panel_label(ax2, "B")

    # Panel C: beta_IC^2 bars from the paper-canonical adjusted spec:
    # DV ~ z(IC) + z(IC)^2 + z(pre-belief) + z(word count), z with ddof=1.
    rows = []
    for tag, c, df in entries:
        sub = df.dropna(subset=["IC", "DV_BeliefChange_Specific",
                                "Pre_Belief_Specific", "OpenendedResponseWordCount"])
        def _z(v):
            v = np.asarray(v, float)
            sd = v.std(ddof=1)
            return (v - v.mean()) / (sd if sd > 0 else 1.0)
        icz = _z(sub.IC.values)
        ic2_z = icz ** 2
        pre_z = _z(sub.Pre_Belief_Specific.values)
        wc_z = _z(sub.OpenendedResponseWordCount.values)
        X = sm.add_constant(np.column_stack([icz, ic2_z, pre_z, wc_z]))
        m_signed = sm.OLS(sub.DV_BeliefChange_Specific.values, X).fit()
        m_abs = sm.OLS(sub.DV_BeliefChange_Specific.abs().values, X).fit()
        rows.append(dict(tag=tag, c=c,
                         b_s=m_signed.params[2], se_s=m_signed.bse[2], p_s=m_signed.pvalues[2],
                         b_a=m_abs.params[2],    se_a=m_abs.bse[2],    p_a=m_abs.pvalues[2]))
        print(f"  Q{tag}: signed beta={m_signed.params[2]:.3f} (p={m_signed.pvalues[2]:.2e}), "
              f"|D| beta={m_abs.params[2]:.3f} (p={m_abs.pvalues[2]:.2e})")
    xs = np.arange(len(rows)); w = 0.35
    tags = [r["tag"] for r in rows]; cs = [r["c"] for r in rows]
    b_s = [r["b_s"] for r in rows]; se_s = [r["se_s"] for r in rows]
    b_a = [r["b_a"] for r in rows]; se_a = [r["se_a"] for r in rows]
    ax3.bar(xs - w / 2, b_s, w, color=cs, alpha=0.75, edgecolor="black",
            lw=0.4, label="Signed $\\beta_{IC^2}$", hatch="")
    ax3.bar(xs + w / 2, b_a, w, color=cs, alpha=0.45, edgecolor="black",
            lw=0.4, label=r"$|\Delta|$ $\beta_{IC^2}$", hatch="//")
    ax3.errorbar(xs - w / 2, b_s, yerr=[1.96 * s for s in se_s], fmt="none",
                 color="black", lw=0.7, capsize=2)
    ax3.errorbar(xs + w / 2, b_a, yerr=[1.96 * s for s in se_a], fmt="none",
                 color="black", lw=0.7, capsize=2)
    bar_span = max(abs(min(b_s + b_a)), abs(max(b_s + b_a))) or 1.0
    txt_off = 0.06 * bar_span
    for i, r in enumerate(rows):
        ps_s = "<.001" if r["p_s"] < .001 else f"{r['p_s']:.3f}"
        ps_a = "<.001" if r["p_a"] < .001 else f"{r['p_a']:.3f}"
        ax3.text(xs[i] - w / 2, r["b_s"] - txt_off, f"{r['b_s']:.2f}\n{ps_s}",
                 ha="center", va="top", fontsize=5)
        ax3.text(xs[i] + w / 2, r["b_a"] - txt_off, f"{r['b_a']:.2f}\n{ps_a}",
                 ha="center", va="top", fontsize=5)
    ax3.set_xticks(xs); ax3.set_xticklabels([f"Q{t}" for t in tags])
    ax3.set_xlabel("Qwen3.5-ORPO checkpoint")
    ax3.set_ylabel(r"$\beta_{IC^2}$ (canonical $z(\mathrm{IC})^2$ scale)")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.legend(frameon=False, fontsize=5.5, loc="lower right")
    ax3.set_title("Effect stability: signed + $|\\Delta|$", fontsize=7, fontweight="bold")
    add_panel_label(ax3, "C")

    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "si_figure5_scoring_stability.pdf", bbox_inches="tight")
    fig.savefig(OUT / "si_figure5_scoring_stability.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}/si_figure5_scoring_stability.pdf and .png")


if __name__ == "__main__":
    main()
