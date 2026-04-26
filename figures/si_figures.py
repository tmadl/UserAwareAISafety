#!/usr/bin/env python3
"""
Generate SI figures for PNAS paper — User-Aware AI Safety.
All figures use real data from Costello, Cheng, and Salvi datasets.
Matches main-paper style from pnas_figures.py.

Outputs:
    paper/PNAS/figures/si_figure1_specification_curve.pdf/png
    paper/PNAS/figures/si_figure2_three_study_replication.pdf/png
    paper/PNAS/figures/si_figure3_ic_distributions.pdf/png
    paper/PNAS/figures/si_figure4_enrichment.pdf/png
    paper/PNAS/figures/si_figure5_scoring_stability.pdf/png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── Global PNAS style (matches main figures) ──────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

BLUE = '#2166AC'
RED = '#B2182B'
GREEN = '#4DAF4A'
GRAY = '#969696'
SALMON = '#F4A582'
PURPLE = '#7570B3'

BASE = '..'
DATA = f'{BASE}/data'
OUT = "./output"


def add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


def load_and_merge(dataset_dir, dv_col, ic_col='IC_openai', key='participantId'):
    ad = pd.read_csv(f'{DATA}/{dataset_dir}/analysis_data.csv')
    cs = pd.read_csv(f'{DATA}/{dataset_dir}/all_complexity_scores.csv')
    df = ad.merge(cs, on=key, how='inner')
    df = df.dropna(subset=[ic_col, dv_col])
    return df


def zscore_ic(df, ic_col='IC_openai'):
    m, s = df[ic_col].mean(), df[ic_col].std()
    df = df.copy()
    df['IC_z'] = (df[ic_col] - m) / s
    df['IC_z2'] = df['IC_z'] ** 2
    return df


def rank_quintiles(df, ic_col='IC_z'):
    df = df.copy()
    df['_rank'] = df[ic_col].rank(method='first')
    df['q'] = pd.qcut(df['_rank'], 5, labels=False)
    return df


# ── Load data ────────────────────────────────────────────────────────────
print("Loading data...")
costello = load_and_merge('costello2024', 'DV_BeliefChange_Specific')
cheng_all = load_and_merge('cheng2006', 'rightorwrong')
cheng3 = cheng_all[cheng_all['study'] == 3].copy()
salvi = load_and_merge('salvi2005', 'opinion_change')
salvi['abs_opinion_change'] = salvi['opinion_change'].abs()
print(f"  Costello: n={len(costello)}, Cheng S3: n={len(cheng3)}, Salvi: n={len(salvi)}")


# ══════════════════════════════════════════════════════════════════════════
# SI FIGURE 1: Specification Curve
# ══════════════════════════════════════════════════════════════════════════

def make_si_figure1():
    print("Generating SI Figure 1: Specification curve...")

    cost_ext = costello.copy()

    # Add extra DV and covariate columns from original publication data
    orig_csv = f'{DATA}/costello2024/Data 8.28.24/AllDataForPublication.PPI.8.28.24.csv'
    try:
        orig = pd.read_csv(orig_csv, low_memory=False)
        pid_level = orig.drop_duplicates(subset=['participantId'], keep='first')
        for col in ['userResponse_combined', 'Education_Cat', 'PercentChange', 'BCTI_Difference']:
            if col in pid_level.columns and col not in cost_ext.columns:
                cost_ext = cost_ext.merge(
                    pid_level[['participantId', col]], on='participantId', how='left')
        if 'userResponse_combined' in cost_ext.columns:
            cost_ext['ttr'] = cost_ext['userResponse_combined'].apply(
                lambda x: len(set(x.lower().split())) / max(len(x.lower().split()), 1)
                if isinstance(x, str) and len(x.strip()) > 0 else np.nan)
        edu_map = {"LessThanHighSchool": 1, "HighSchool": 2, "SomeCollege": 3,
                   "Associate": 3, "Bachelors": 4, "Masters": 5, "JD/MD": 6, "PhD": 6}
        if 'Education_Cat' in cost_ext.columns:
            cost_ext['edu_num'] = cost_ext['Education_Cat'].map(edu_map)
    except Exception:
        pass

    specs = []

    # Primary DV only — PercentChange has numerical issues, BCTI is a different construct
    dv_options = {"DV_BeliefChange_Specific": "Belief change"}

    # Single IC measure used throughout the paper
    ic_options = {"IC_openai": "IC"}

    cov_sets = {
        "none": [],
        "+pre": ["Pre_Belief_Specific"],
        "+pre+wc": ["Pre_Belief_Specific", "OpenendedResponseWordCount"],
    }
    if "ttr" in cost_ext.columns:
        cov_sets["+pre+wc+ttr"] = ["Pre_Belief_Specific", "OpenendedResponseWordCount", "ttr"]
    if "edu_num" in cost_ext.columns and "ttr" in cost_ext.columns:
        cov_sets["+pre+wc+ttr+edu"] = ["Pre_Belief_Specific", "OpenendedResponseWordCount",
                                        "ttr", "edu_num"]

    study_opts = {"all": None, "S1": 1.0, "S2": 2.0, "S3": 3.0}
    outlier_opts = {"none": None, "3SD": 3.0, "2SD": 2.0}

    for dv_col, dv_label in dv_options.items():
        for ic_col, ic_label in ic_options.items():
            for cov_label, covs in cov_sets.items():
                for study_label, study_num in study_opts.items():
                    for out_label, out_sd in outlier_opts.items():
                        sub = cost_ext.copy()
                        if study_num is not None:
                            sub = sub[sub["StudyNumber"] == study_num]

                        req = [dv_col, ic_col] + covs
                        sub = sub.dropna(subset=[c for c in req if c in sub.columns])
                        if len(sub) < 50:
                            continue

                        if out_sd is not None:
                            dv_vals = sub[dv_col]
                            mu, sigma = dv_vals.mean(), dv_vals.std()
                            sub = sub[np.abs(dv_vals - mu) < out_sd * sigma]

                        if len(sub) < 50:
                            continue

                        y = sub[dv_col].values
                        ic_raw = sub[ic_col].values
                        ic_z = (ic_raw - ic_raw.mean()) / ic_raw.std()
                        ic_z2 = ic_z ** 2

                        preds = [ic_z, ic_z2]
                        for cv in covs:
                            if cv in sub.columns:
                                v = sub[cv].values
                                preds.append((v - v.mean()) / max(v.std(), 1e-8))

                        X = sm.add_constant(np.column_stack(preds))
                        try:
                            m = sm.OLS(y, X).fit()
                            b_sq = m.params[2]
                            p_sq = m.pvalues[2]
                            se_sq = m.bse[2]
                            if np.isnan(b_sq) or np.isinf(b_sq):
                                continue
                        except Exception:
                            continue

                        specs.append({
                            "dv": dv_label, "ic": ic_label, "covs": cov_label,
                            "study": study_label, "outliers": out_label,
                            "n": len(sub), "b_sq": b_sq, "se_sq": se_sq,
                            "p_sq": p_sq, "r2": m.rsquared,
                        })

    df_specs = pd.DataFrame(specs)
    n_total = len(df_specs)
    print(f"  Total specifications: {n_total}")

    # Sort by effect size
    df_specs = df_specs.sort_values("b_sq").reset_index(drop=True)
    is_neg = df_specs["b_sq"] < 0
    is_sig = df_specs["p_sq"] < 0.05
    pct_neg = is_neg.mean() * 100
    pct_sig = is_sig.mean() * 100
    print(f"  Correct sign: {pct_neg:.0f}%, Significant: {pct_sig:.0f}%")

    fig = plt.figure(figsize=(7.08, 4.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # Top panel: sorted effect sizes with CIs
    xs = np.arange(n_total)
    for i in range(n_total):
        b = df_specs.iloc[i]["b_sq"]
        se = df_specs.iloc[i]["se_sq"]
        color = BLUE if b < 0 else SALMON
        alpha = 0.8 if df_specs.iloc[i]["p_sq"] < 0.05 else 0.3
        ax_top.plot([i, i], [b - 1.96 * se, b + 1.96 * se],
                    color=color, linewidth=0.4, alpha=alpha * 0.5)
        ax_top.plot(i, b, '.', color=color, markersize=1.2, alpha=alpha)

    ax_top.axhline(0, color='black', linewidth=0.5)
    med = df_specs["b_sq"].median()
    ax_top.axhline(med, color=GRAY, linewidth=0.5, linestyle='--', alpha=0.7)
    ax_top.text(n_total * 0.02, med + 0.3, f'Median = {med:.2f}',
                fontsize=5.5, color=GRAY)
    ax_top.text(0.97, 0.95, f'{pct_neg:.0f}% predicted sign\n{pct_sig:.0f}% significant',
                transform=ax_top.transAxes, fontsize=6.5, va='top', ha='right',
                fontweight='bold')
    ax_top.set_ylabel('Quadratic $\\beta_{IC^2}$')
    ax_top.tick_params(axis='x', labelbottom=False)
    add_panel_label(ax_top, 'A', x=-0.06, y=1.05)

    # Bottom panel: indicator matrix (only dimensions that vary)
    cat_map = {}
    for cat_name, col, color in [
        ('Covariates', 'covs', BLUE),
        ('Study', 'study', RED),
        ('Outliers', 'outliers', GRAY),
    ]:
        options = list(df_specs[col].unique())
        if len(options) > 1:
            cat_map[cat_name] = (col, options)
    cat_colors = {'Covariates': BLUE, 'Study': RED, 'Outliers': GRAY}

    row_offset = 0
    ytick_pos = []
    ytick_lab = []
    for cat_name, (col, options) in cat_map.items():
        for j, opt in enumerate(options):
            row = row_offset + j
            ytick_pos.append(row)
            ytick_lab.append(opt)
            mask = df_specs[col] == opt
            idxs = np.where(mask.values)[0]
            # Draw solid colored blocks for active specs
            for si in idxs:
                ax_bot.add_patch(plt.Rectangle((si - 0.4, row - 0.4), 0.8, 0.8,
                    facecolor=cat_colors[cat_name], alpha=0.7, linewidth=0))
        # Category label on right
        mid = row_offset + len(options) / 2 - 0.5
        ax_bot.text(n_total + n_total * 0.02, mid, cat_name, fontsize=5,
                    va='center', fontweight='bold', color=cat_colors[cat_name],
                    clip_on=False)
        if row_offset > 0:
            ax_bot.axhline(row_offset - 0.5, color=GRAY, linewidth=0.3, alpha=0.5)
        row_offset += len(options)

    ax_bot.set_yticks(ytick_pos)
    ax_bot.set_yticklabels(ytick_lab, fontsize=5)
    ax_bot.set_ylim(-1, row_offset)
    ax_bot.set_xlim(-1, n_total + 1)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel('Specification (sorted by effect size)')
    add_panel_label(ax_bot, 'B', x=-0.06, y=1.05)

    fig.savefig(f'{OUT}/si_figure1_specification_curve.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/si_figure1_specification_curve.png', bbox_inches='tight')
    plt.close(fig)
    print("  SI Figure 1 saved.")


# ══════════════════════════════════════════════════════════════════════════
# SI FIGURE 2: Three-Study Replication
# ══════════════════════════════════════════════════════════════════════════

def make_si_figure2():
    print("Generating SI Figure 2: Three-study replication...")

    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.6))

    study_colors = {1: '#2166AC', 2: '#4393C3', 3: '#92C5DE'}

    for i_s, (ax, study_num) in enumerate(zip(axes, [1, 2, 3])):
        sub = costello[costello['StudyNumber'] == study_num].copy()
        sub = zscore_ic(sub)
        x = sub['IC_z'].values
        y = sub['DV_BeliefChange_Specific'].values
        color = study_colors[study_num]

        # Scatter
        jx = x + np.random.RandomState(i_s).normal(0, 0.03, len(x))
        jy = y + np.random.RandomState(i_s + 10).normal(0, 0.3, len(y))
        ax.scatter(jx, jy, s=3, alpha=0.08, color=color, edgecolors='none',
                   rasterized=True)

        # Quadratic fit + bootstrap CI
        xgrid = np.linspace(x.min(), x.max(), 200)
        preds = np.zeros((500, 200))
        rng = np.random.RandomState(42)
        for b in range(500):
            idx = rng.choice(len(x), len(x), replace=True)
            try:
                c = np.polyfit(x[idx], y[idx], 2)
                preds[b] = np.polyval(c, xgrid)
            except:
                preds[b] = np.nan
        lo = np.nanpercentile(preds, 2.5, axis=0)
        hi = np.nanpercentile(preds, 97.5, axis=0)
        med = np.nanpercentile(preds, 50, axis=0)
        ax.fill_between(xgrid, lo, hi, alpha=0.2, color=color, linewidth=0)
        ax.plot(xgrid, med, color=color, linewidth=1.5)

        # Quintile means
        sub_q = rank_quintiles(sub)
        for q in range(5):
            sq = sub_q[sub_q['q'] == q]['DV_BeliefChange_Specific']
            xm = sub_q[sub_q['q'] == q]['IC_z'].mean()
            ym = sq.mean()
            ci = 1.96 * sq.std() / np.sqrt(len(sq))
            ax.errorbar(xm, ym, yerr=ci, fmt='o', color=color, markersize=4,
                       markerfacecolor='white', markeredgewidth=1,
                       markeredgecolor=color, linewidth=0.8, capsize=2,
                       capthick=0.5, zorder=5)

        # Stats
        X = sm.add_constant(np.column_stack([x, x**2]))
        m = sm.OLS(y, X).fit()
        p_str = 'p < .001' if m.pvalues[2] < 0.001 else f'p = {m.pvalues[2]:.3f}'
        ax.text(0.97, 0.97,
                f'$\\beta_{{IC^2}}$ = {m.params[2]:.2f}\n{p_str}\nN = {len(sub)}',
                transform=ax.transAxes, fontsize=6, va='top', ha='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

        ax.set_xlabel('Integrative complexity (z-scored)')
        ax.set_ylabel('Belief change (0\u2013100)')
        ax.set_title(f'Study {study_num}', fontsize=8, fontweight='bold')
        add_panel_label(ax, chr(64 + i_s + 1))

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f'{OUT}/si_figure2_three_study_replication.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/si_figure2_three_study_replication.png', bbox_inches='tight')
    plt.close(fig)
    print("  SI Figure 2 saved.")


# ══════════════════════════════════════════════════════════════════════════
# SI FIGURE 3: IC Score Distributions
# ══════════════════════════════════════════════════════════════════════════

def make_si_figure3():
    print("Generating SI Figure 3: IC score distributions...")

    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.4))

    datasets = [
        (costello, 'Costello', BLUE),
        (cheng3, 'Cheng Study 3', RED),
        (salvi, 'Salvi', GREEN),
    ]

    # Panel A: Overlaid histograms
    ax = axes[0]
    bins = np.arange(0.5, 8.0, 0.5)
    for df, name, color in datasets:
        ic = df['IC_openai'].values
        ax.hist(ic, bins=bins, alpha=0.35, color=color, label=name,
                density=True, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('IC score')
    ax.set_ylabel('Density')
    ax.set_title('IC score distributions', fontsize=7, fontweight='bold')
    ax.legend(frameon=False, fontsize=5.5)
    add_panel_label(ax, 'A')

    # Panel B: KDE overlay — use different linestyles so overlapping curves are visible
    ax = axes[1]
    linestyles = ['-', '-', '--']
    linewidths = [1.5, 1.5, 1.8]
    for (df, name, color), ls, lw in zip(datasets, linestyles, linewidths):
        ic = df['IC_openai'].values
        xgrid = np.linspace(0.5, 7.5, 300)
        try:
            kde = stats.gaussian_kde(ic, bw_method=0.3)
            ax.plot(xgrid, kde(xgrid), color=color, linewidth=lw, linestyle=ls, label=name)
            ax.fill_between(xgrid, kde(xgrid), alpha=0.1, color=color)
        except Exception:
            pass
    ax.set_xlabel('IC score')
    ax.set_ylabel('Density')
    ax.set_title('Kernel density estimates', fontsize=7, fontweight='bold')
    ax.legend(frameon=False, fontsize=5.5)
    add_panel_label(ax, 'B')

    # Panel C: Summary stats + KS tests
    ax = axes[2]
    ax.axis('off')

    lines = []
    lines.append(('Dataset', 'N', 'Mean', 'SD', 'Range'))
    for df, name, _ in datasets:
        ic = df['IC_openai'].values
        lines.append((name, f'{len(df):,}', f'{ic.mean():.2f}', f'{ic.std():.2f}',
                      f'[{ic.min():.1f}, {ic.max():.1f}]'))

    # KS tests
    ks_cs, p_cs = stats.ks_2samp(costello['IC_openai'], salvi['IC_openai'])
    ks_cc, p_cc = stats.ks_2samp(costello['IC_openai'], cheng3['IC_openai'])
    ks_sc, p_sc = stats.ks_2samp(cheng3['IC_openai'], salvi['IC_openai'])

    table = ax.table(
        cellText=[list(r) for r in lines[1:]],
        colLabels=lines[0],
        loc='upper center',
        cellLoc='center',
        colColours=['#f0f0f0'] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.5)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)

    ks_text = (f'KS tests:\n'
               f'Costello vs Salvi: D={ks_cs:.3f}, '
               f'{"p < .001" if p_cs < .001 else f"p = {p_cs:.3f}"}\n'
               f'Costello vs Cheng: D={ks_cc:.3f}, '
               f'{"p < .001" if p_cc < .001 else f"p = {p_cc:.3f}"}\n'
               f'Cheng vs Salvi: D={ks_sc:.3f}, '
               f'{"p < .001" if p_sc < .001 else f"p = {p_sc:.3f}"}')
    ax.text(0.5, 0.15, ks_text, transform=ax.transAxes, fontsize=5.5,
            va='top', ha='center', family='monospace')
    ax.set_title('Distributional comparisons', fontsize=7, fontweight='bold')
    add_panel_label(ax, 'C', x=-0.05)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(f'{OUT}/si_figure3_ic_distributions.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/si_figure3_ic_distributions.png', bbox_inches='tight')
    plt.close(fig)
    print("  SI Figure 3 saved.")


# ══════════════════════════════════════════════════════════════════════════
# SI FIGURE 4: Enrichment Analysis
# ══════════════════════════════════════════════════════════════════════════

def make_si_figure4():
    print("Generating SI Figure 4: Enrichment analysis...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.08, 2.8))

    cost = costello.copy()
    cost['backlash'] = (cost['DV_BeliefChange_Specific'] < 0).astype(int)
    cost['large_change'] = (cost['DV_BeliefChange_Specific'] >= 20).astype(int)
    cost = rank_quintiles(zscore_ic(cost))

    total_n = len(cost)
    total_lc = cost['large_change'].sum()
    total_bl = cost['backlash'].sum()

    # Panel A: Retention curves
    thresholds = list(range(5))  # drop Q0, Q0-1, Q0-2, Q0-3, Q0-4
    x_labels = ['None', 'Q1', 'Q1\u2013Q2', 'Q1\u2013Q3', 'Q1\u2013Q4']
    lc_ret, bl_excl, n_ret = [], [], []

    for drop_up_to in range(5):
        if drop_up_to == 0:
            ret = cost
        else:
            ret = cost[cost['q'] >= drop_up_to]
        n_ret.append(len(ret) / total_n)
        lc_ret.append(ret['large_change'].sum() / max(total_lc, 1))
        bl_excl.append(1 - ret['backlash'].sum() / max(total_bl, 1))

    xs = np.arange(5)
    lc_arr = np.array(lc_ret)
    bl_arr = np.array(bl_excl)

    ax1.fill_between(xs, bl_arr, lc_arr,
                     where=(lc_arr >= bl_arr),
                     alpha=0.15, color=GREEN, interpolate=True)
    ax1.plot(xs, lc_ret, 'o-', color=BLUE, linewidth=1.5, markersize=5,
             label='Large changers retained')
    ax1.plot(xs, bl_excl, 's--', color=RED, linewidth=1.5, markersize=5,
             label='Backlash excluded')
    ax1.plot(xs, n_ret, '^:', color=GRAY, linewidth=1.0, markersize=4,
             label='Sample retained')

    # Annotate Drop Q1
    ax1.plot(1, lc_ret[1], 'D', color=BLUE, markersize=7, zorder=6,
             markeredgecolor='black', markeredgewidth=0.5)
    ax1.annotate(f'Retains {lc_ret[1]:.0%} of benefit\nRemoves {bl_excl[1]:.0%} of backlash',
                 xy=(1, lc_ret[1]), xytext=(2.2, 0.92), fontsize=5.5,
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.7),
                 bbox=dict(facecolor='white', edgecolor=GRAY,
                          boxstyle='round,pad=0.3', linewidth=0.5))

    ax1.set_xticks(xs)
    ax1.set_xticklabels(x_labels, fontsize=6)
    ax1.set_xlabel('IC quintile(s) excluded')
    ax1.set_ylabel('Proportion')
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9,
               edgecolor=GRAY, fontsize=5.5)
    ax1.set_ylim(-0.05, 1.10)
    ax1.set_title('Enrichment: IC-based screening', fontsize=7, fontweight='bold')
    add_panel_label(ax1, 'A')

    # Panel B: Backlash rate by quintile
    quintile_bl = []
    quintile_lc = []
    for q in range(5):
        sq = cost[cost['q'] == q]
        quintile_bl.append(sq['backlash'].mean())
        quintile_lc.append(sq['large_change'].mean())

    xs2 = np.arange(5)
    width = 0.35
    bars_bl = ax2.bar(xs2 - width/2, quintile_bl, width, color=RED, alpha=0.7,
                      label='Backlash rate')
    bars_lc = ax2.bar(xs2 + width/2, quintile_lc, width, color=BLUE, alpha=0.7,
                      label='Large change rate')

    ax2.set_xticks(xs2)
    ax2.set_xticklabels(['Q1\nLow IC', 'Q2', 'Q3', 'Q4', 'Q5\nHigh IC'], fontsize=6)
    ax2.set_xlabel('IC Quintile')
    ax2.set_ylabel('Proportion')
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9,
               edgecolor=GRAY, fontsize=5.5)
    ax2.set_title('Backlash vs. benefit by IC quintile', fontsize=7, fontweight='bold')

    # Add percentage labels on bars
    for bar in bars_bl:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.0%}', ha='center', va='bottom', fontsize=5)
    for bar in bars_lc:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.0%}', ha='center', va='bottom', fontsize=5)

    add_panel_label(ax2, 'B')

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f'{OUT}/si_figure4_enrichment.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/si_figure4_enrichment.png', bbox_inches='tight')
    plt.close(fig)
    print("  SI Figure 4 saved.")


# ══════════════════════════════════════════════════════════════════════════
# SI FIGURE 5: Scoring Run Stability
# ══════════════════════════════════════════════════════════════════════════

def make_si_figure5():
    print("Generating SI Figure 5: Scoring run stability...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.08, 2.8))

    datasets_runs = {
        'Costello (all)': f'{DATA}/costello2024/checkpoints/ic_openai_all_runs.csv',
        'Costello (initial)': f'{DATA}/costello2024/checkpoints/ic_openai_initial_runs.csv',
        'Cheng (all)': f'{DATA}/cheng2006/checkpoints/ic_openai_all_runs.csv',
        'Cheng (initial)': f'{DATA}/cheng2006/checkpoints/ic_openai_initial_runs.csv',
        'Salvi (all)': f'{DATA}/salvi2005/checkpoints/ic_openai_all_runs.csv',
        'Salvi (initial)': f'{DATA}/salvi2005/checkpoints/ic_openai_initial_runs.csv',
    }

    colors_runs = {
        'Costello (all)': BLUE, 'Costello (initial)': BLUE,
        'Cheng (all)': RED, 'Cheng (initial)': RED,
        'Salvi (all)': GREEN, 'Salvi (initial)': GREEN,
    }
    linestyles = {
        'Costello (all)': '-', 'Costello (initial)': '--',
        'Cheng (all)': '-', 'Cheng (initial)': '--',
        'Salvi (all)': '-', 'Salvi (initial)': '--',
    }

    # Panel A: Correlation of n-run average with 10-run average
    n_runs_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for name, path in datasets_runs.items():
        try:
            runs_df = pd.read_csv(path)
        except FileNotFoundError:
            continue

        run_cols = [f'run_{i}' for i in range(10)]
        avail_cols = [c for c in run_cols if c in runs_df.columns]
        if len(avail_cols) < 10:
            continue

        runs_mat = runs_df[avail_cols].values
        ref = np.nanmean(runs_mat, axis=1)
        valid = ~np.isnan(ref)

        corrs = []
        for n in n_runs_test:
            avg_n = np.nanmean(runs_mat[:, :n], axis=1)
            mask = valid & ~np.isnan(avg_n)
            if mask.sum() > 10:
                r, _ = stats.pearsonr(avg_n[mask], ref[mask])
                corrs.append(r)
            else:
                corrs.append(np.nan)

        ax1.plot(n_runs_test, corrs, marker='o', markersize=3,
                 color=colors_runs[name], linestyle=linestyles[name],
                 linewidth=1.0, label=name, alpha=0.8)

    ax1.set_xlabel('Number of runs averaged')
    ax1.set_ylabel('Pearson r with 10-run average')
    ax1.set_ylim(0.97, 1.001)
    ax1.set_xticks(range(1, 10))
    ax1.axhline(0.993, color=GRAY, linewidth=0.5, linestyle=':', alpha=0.5)
    ax1.text(5, 0.9935, 'r = .993', fontsize=5, color=GRAY)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9,
               edgecolor=GRAY, fontsize=4.5, ncol=2)
    ax1.set_title('Convergence with number of runs', fontsize=7, fontweight='bold')
    add_panel_label(ax1, 'A')

    # Panel B: Within-text SD distribution (one representative dataset)
    try:
        runs_all = pd.read_csv(f'{DATA}/costello2024/checkpoints/ic_openai_all_runs.csv')
        run_cols = [f'run_{i}' for i in range(10)]
        runs_mat = runs_all[run_cols].values
        within_sd = np.nanstd(runs_mat, axis=1, ddof=1)
        # Replace NaN SD (single value) with 0
        within_sd = np.nan_to_num(within_sd, nan=0.0)

        pct_zero = np.mean(within_sd == 0) * 100

        ax2.hist(within_sd, bins=np.arange(0, 1.05, 0.05), color=BLUE, alpha=0.7,
                 edgecolor='white', linewidth=0.3)
        ax2.axvline(np.mean(within_sd), color=RED, linewidth=1.0, linestyle='--',
                    label=f'Mean SD = {np.mean(within_sd):.3f}')
        ax2.text(0.97, 0.95,
                 f'{pct_zero:.0f}% zero variance\nN = {len(within_sd):,}',
                 transform=ax2.transAxes, fontsize=6, va='top', ha='right',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax2.set_xlabel('Within-text SD across 10 runs')
        ax2.set_ylabel('Count')
        ax2.legend(loc='upper center', frameon=False, fontsize=6)
        ax2.set_title('IC scoring variability (Costello)', fontsize=7, fontweight='bold')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Data not available:\n{e}', transform=ax2.transAxes,
                 ha='center', va='center', fontsize=7)
        ax2.set_title('IC scoring variability', fontsize=7, fontweight='bold')

    add_panel_label(ax2, 'B')

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f'{OUT}/si_figure5_scoring_stability.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/si_figure5_scoring_stability.png', bbox_inches='tight')
    plt.close(fig)
    print("  SI Figure 5 saved.")


# ── Run all ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    make_si_figure1()
    make_si_figure2()
    make_si_figure3()
    make_si_figure4()
    make_si_figure5()
    print(f'\nAll SI figures saved to {OUT}/')
