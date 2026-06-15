"""Regenerate figures/si_figure4_enrichment with the caption's adverse-movement
definition (DV <= -5) replacing the stale backlash (<0) definition.

Identical styling to scripts/figures/si_figures_q400.py make_fig4 except:
  - adverse movement = DV_BeliefChange_Specific <= -5 (was: < 0)
  - panel A annotation shows one-decimal 86.3% / 24.4% operating point
  - legend labels say "Adverse movement (<= -5)" instead of "Backlash (signed<0)"
  - panel C note reference corrected: Note 10 -> Note 9 (sec:absdv)
  - output -> paper/PNAS_Nexus_v3/figures
"""
import sys
from pathlib import Path
sys.path.insert(0, "/mnt/workvm/UserAwareAISafety/scripts/figures")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import PNAS_RC, BLUE, RED, GREEN, GRAY, COL2, load_costello_q400

plt.rcParams.update(PNAS_RC)
OUT = Path("/mnt/workvm/UserAwareAISafety/paper/PNAS_Nexus_v3/figures")


def add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


def main():
    cost = load_costello_q400().rename(columns={"IC_q4": "IC"})
    print(f"Costello N = {len(cost)}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(COL2, 2.9))
    c = cost.copy()
    c["adverse"] = (c.DV_BeliefChange_Specific <= -5).astype(int)
    c["large_change"] = (c.DV_BeliefChange_Specific >= 20).astype(int)
    c["absdv"] = c.DV_BeliefChange_Specific.abs()
    c["_r"] = c.IC.rank(method="first")
    c["q"] = pd.qcut(c._r, 5, labels=False)
    tot_n = len(c); tot_lc = c.large_change.sum(); tot_am = c.adverse.sum()

    x_labels = ["None", "Q1", "Q1–Q2", "Q1–Q3", "Q1–Q4"]
    lc_ret, am_excl, n_ret = [], [], []
    for drop in range(5):
        ret = c if drop == 0 else c[c.q >= drop]
        n_ret.append(len(ret) / tot_n)
        lc_ret.append(ret.large_change.sum() / max(tot_lc, 1))
        am_excl.append(1 - ret.adverse.sum() / max(tot_am, 1))
    print("drop-Q1 operating point: LC retained {:.1%}, adverse excluded {:.1%}".format(
        lc_ret[1], am_excl[1]))

    xs = np.arange(5)
    lc_a = np.array(lc_ret); am_a = np.array(am_excl)
    ax1.fill_between(xs, am_a, lc_a, where=(lc_a >= am_a), alpha=0.15, color=GREEN, interpolate=True)
    ax1.plot(xs, lc_ret, "o-", color=BLUE, lw=1.5, ms=5, label="Large changers retained")
    ax1.plot(xs, am_excl, "s--", color=RED, lw=1.5, ms=5, label="Adverse movement excluded")
    ax1.plot(xs, n_ret, "^:", color=GRAY, lw=1, ms=4, label="Sample retained")
    ax1.plot(1, lc_ret[1], "D", color=BLUE, ms=7, zorder=6, mec="black", mew=0.5)
    ax1.annotate(f"Retains {lc_ret[1]:.1%} of large changes\nExcludes {am_excl[1]:.1%} of adverse movement",
                 xy=(1, lc_ret[1]), xytext=(1.7, 0.97), fontsize=5.5,
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
                 bbox=dict(facecolor="white", edgecolor=GRAY, boxstyle="round,pad=0.3", lw=0.5))
    ax1.set_xticks(xs); ax1.set_xticklabels(x_labels, fontsize=6)
    ax1.set_xlabel("IC quintile(s) excluded")
    ax1.set_ylabel("Proportion")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor=GRAY, fontsize=5)
    ax1.set_ylim(-0.05, 1.10)
    ax1.set_title("Signed-DV enrichment: IC screening", fontsize=7, fontweight="bold", pad=12)
    add_panel_label(ax1, "A", x=-0.14, y=1.18)

    q_am = [c[c.q == q].adverse.mean() for q in range(5)]
    q_lc = [c[c.q == q].large_change.mean() for q in range(5)]
    print("per-quintile adverse(<=-5):", [f"{v:.1%}" for v in q_am])
    print("per-quintile large-change :", [f"{v:.1%}" for v in q_lc])
    xs2 = np.arange(5); w = 0.35
    b1 = ax2.bar(xs2 - w / 2, q_am, w, color=RED, alpha=0.7, label=r"Adverse movement ($\leq-5$)")
    b2 = ax2.bar(xs2 + w / 2, q_lc, w, color=BLUE, alpha=0.7, label=r"Large signed change ($\geq$20)")
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(["Q1\nLow IC", "Q2", "Q3", "Q4", "Q5\nHigh IC"], fontsize=6)
    ax2.set_xlabel("IC quintile"); ax2.set_ylabel("Proportion")
    ax2.legend(loc="lower center", frameon=True, framealpha=0.9, edgecolor=GRAY, fontsize=5.5)
    ax2.set_title("Signed-DV outcomes by IC quintile", fontsize=7, fontweight="bold", pad=12)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + 0.005, f"{h:.0%}",
                 ha="center", va="bottom", fontsize=5)
    add_panel_label(ax2, "B", x=-0.14, y=1.18)

    # Panel C: |Delta| by IC quintile (unchanged from previous version except
    # the cross-reference: sec:absdv is Note 9)
    q_ic = [c[c.q == q].IC.mean() for q in range(5)]
    q_abs_mean = [c[c.q == q].absdv.mean() for q in range(5)]
    q_abs_sem = [c[c.q == q].absdv.sem() * 1.96 for q in range(5)]
    ax3.errorbar(xs2, q_abs_mean, yerr=q_abs_sem, fmt="o-", color=BLUE, lw=1.5, ms=5,
                 mfc="white", mew=1.2, capsize=3)
    coef = np.polyfit(q_ic, q_abs_mean, 2)
    xgrid = np.linspace(min(q_ic), max(q_ic), 100)
    ax3_x_from_ic = np.interp(xgrid, q_ic, xs2)
    ax3.plot(ax3_x_from_ic, np.polyval(coef, xgrid), color=BLUE, lw=0.8, ls="--", alpha=0.7)
    for i, (x_i, v) in enumerate(zip(xs2, q_abs_mean)):
        ax3.text(x_i, v + q_abs_sem[i] + 0.5, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=5)
    ax3.set_xticks(xs2)
    ax3.set_xticklabels(["Q1\nLow IC", "Q2", "Q3", "Q4", "Q5\nHigh IC"], fontsize=6)
    ax3.set_xlabel("IC quintile")
    ax3.set_ylabel(r"$|\Delta\mathrm{belief}|$ (0--100)")
    ax3.set_title(r"Engagement magnitude $|\Delta|$: inverted-U", fontsize=7, fontweight="bold", pad=12)
    ax3.text(0.5, 0.03,
             r"$\beta_{IC^2}=-3.53$, $p<.001$" + "\n" + r"BF$_{10}=671$ (Note 9)",
             transform=ax3.transAxes, fontsize=5.5, va="bottom", ha="center",
             bbox=dict(facecolor="white", edgecolor=GRAY, boxstyle="round,pad=0.25",
                       lw=0.4, alpha=0.9))
    add_panel_label(ax3, "C", x=-0.14, y=1.18)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "si_figure4_enrichment.pdf", bbox_inches="tight")
    fig.savefig(OUT / "si_figure4_enrichment.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}/si_figure4_enrichment.pdf and .png")


if __name__ == "__main__":
    main()
