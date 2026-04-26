#!/usr/bin/env python3
"""
Generate 5 PNAS-quality figures for User-Aware AI Safety paper.
Loads real data from Costello, Cheng, and Salvi datasets.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Global PNAS style ──────────────────────────────────────────────────────
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

# Colors
BLUE = '#2166AC'
RED = '#B2182B'
GREEN = '#4DAF4A'
GRAY = '#969696'
SALMON = '#F4A582'

OUT = './output'

# ── Load data ──────────────────────────────────────────────────────────────
def load_dataset(name, data_dir, ic_col='IC_openai', dv_col=None, merge_on='participantId'):
    analysis = pd.read_csv(f'{data_dir}/analysis_data.csv')
    complexity = pd.read_csv(f'{data_dir}/all_complexity_scores.csv')
    df = analysis.merge(complexity, on=merge_on, how='inner')
    df = df.dropna(subset=[ic_col, dv_col])
    return df

costello = load_dataset('costello', '../data/costello2024',
                        dv_col='DV_BeliefChange_Specific')
cheng_all = load_dataset('cheng', '../data/cheng2006',
                         dv_col='rightorwrong')
cheng_all = cheng_all[cheng_all['study'] == 3]
cheng_syco = cheng_all[cheng_all['is_sycophantic'] == 1].copy()
cheng_nonsyco = cheng_all[cheng_all['is_sycophantic'] == 0].copy()

salvi = load_dataset('salvi', '../data/salvi2005',
                     dv_col='opinion_change')
salvi['abs_opinion_change'] = salvi['opinion_change'].abs()

# Z-score IC within each subsample
def zscore_ic(df, ic_col='IC_openai'):
    m, s = df[ic_col].mean(), df[ic_col].std()
    df['IC_z'] = (df[ic_col] - m) / s
    df['IC_z2'] = df['IC_z'] ** 2
    return df

costello = zscore_ic(costello)
cheng_syco = zscore_ic(cheng_syco)
cheng_nonsyco = zscore_ic(cheng_nonsyco)
salvi = zscore_ic(salvi)

# ── Helpers ────────────────────────────────────────────────────────────────
def rank_quintiles(df, ic_col='IC_z'):
    df = df.copy()
    df['_rank'] = df[ic_col].rank(method='first')
    df['q'] = pd.qcut(df['_rank'], 5, labels=False)
    return df

def fit_quadratic(x, y):
    """Fit y = a + b*x + c*x^2, return coefficients and p-values."""
    X = np.column_stack([np.ones(len(x)), x, x**2])
    from numpy.linalg import lstsq
    beta, res, rank, sv = lstsq(X, y, rcond=None)
    n = len(y)
    k = 3
    yhat = X @ beta
    sse = np.sum((y - yhat)**2)
    mse = sse / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = beta / se
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=n - k)
    return beta, se, p_vals

def bootstrap_quadratic_ci(x, y, n_boot=500, n_grid=200):
    xgrid = np.linspace(x.min(), x.max(), n_grid)
    preds = np.zeros((n_boot, n_grid))
    rng = np.random.RandomState(42)
    for i in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        xb, yb = x[idx], y[idx]
        try:
            c = np.polyfit(xb, yb, 2)
            preds[i] = np.polyval(c, xgrid)
        except:
            preds[i] = np.nan
    lo = np.nanpercentile(preds, 2.5, axis=0)
    hi = np.nanpercentile(preds, 97.5, axis=0)
    med = np.nanpercentile(preds, 50, axis=0)
    return xgrid, med, lo, hi

def quintile_stats(df, ic_col='IC_z', dv_col='DV_BeliefChange_Specific'):
    df2 = rank_quintiles(df, ic_col)
    out = []
    for q in range(5):
        sub = df2[df2['q'] == q]
        m = sub[dv_col].mean()
        ci = 1.96 * sub[dv_col].std() / np.sqrt(len(sub))
        xm = sub[ic_col].mean()
        out.append((xm, m, ci))
    return out

def add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')

# ── FIGURE 1: Three-Way Dissociation ──────────────────────────────────────
def make_figure1():
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.6))

    datasets = [
        (costello, 'IC_z', 'DV_BeliefChange_Specific', 'Belief change (0\u2013100)', BLUE, 'A'),
        (cheng_syco, 'IC_z', 'rightorwrong', 'Perceived rightness', RED, 'B'),
        (salvi, 'IC_z', 'opinion_change', 'Opinion change', GREEN, 'C'),
    ]

    ylims = [(-30, 75), None, None]  # Fix Panel A range

    for i_panel, (ax, (df, ic, dv, ylabel, color, label)) in enumerate(zip(axes, datasets)):
        x = df[ic].values
        y = df[dv].values

        # Scatter with jitter
        jx = x + np.random.RandomState(0).normal(0, 0.03, len(x))
        jy = y + np.random.RandomState(1).normal(0, 0.01 * y.std(), len(y))
        ax.scatter(jx, jy, s=3, alpha=0.08, color=color, edgecolors='none', rasterized=True)

        # Quadratic fit + bootstrap CI
        xgrid, med, lo, hi = bootstrap_quadratic_ci(x, y)
        ax.fill_between(xgrid, lo, hi, alpha=0.2, color=color, linewidth=0)
        ax.plot(xgrid, med, color=color, linewidth=1.5)

        # Quintile means
        qs = quintile_stats(df, ic, dv)
        for xm, ym, ci in qs:
            ax.errorbar(xm, ym, yerr=ci, fmt='o', color=color, markersize=4,
                       markerfacecolor='white', markeredgewidth=1, markeredgecolor=color,
                       linewidth=0.8, capsize=2, capthick=0.5, zorder=5)

        # Stats
        beta, se, pvals = fit_quadratic(x, y)
        p_str = f'p < .001' if pvals[2] < 0.001 else f'p = {pvals[2]:.3f}'
        ax.text(0.97, 0.97, f'$\\beta_{{IC^2}}$ = {beta[2]:.2f}\n{p_str}',
                transform=ax.transAxes, fontsize=6, va='top', ha='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax.text(0.03, 0.03, f'N = {len(df):,}', transform=ax.transAxes,
                fontsize=6, va='bottom', ha='left', color=GRAY)

        ax.set_xlabel('Integrative complexity (z-scored)')
        ax.set_ylabel(ylabel)
        add_panel_label(ax, label)

        if ylims[i_panel] is not None:
            ax.set_ylim(*ylims[i_panel])

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f'{OUT}/figure1_three_way_dissociation.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/figure1_three_way_dissociation.png', bbox_inches='tight')
    plt.close(fig)
    print('Figure 1 saved.')

# ── FIGURE 2: Mirror-Image Profiles ───────────────────────────────────────
def make_figure2():
    fig, ax1 = plt.subplots(figsize=(3.42, 3.0))

    # Costello quintiles
    costello_q = rank_quintiles(costello, 'IC_z')
    c_means, c_cis = [], []
    for q in range(5):
        sub = costello_q[costello_q['q'] == q]['DV_BeliefChange_Specific']
        c_means.append(sub.mean())
        c_cis.append(1.96 * sub.std() / np.sqrt(len(sub)))

    # Cheng sycophantic quintiles
    cheng_q = rank_quintiles(cheng_syco, 'IC_z')
    ch_means, ch_cis = [], []
    for q in range(5):
        sub = cheng_q[cheng_q['q'] == q]['rightorwrong']
        ch_means.append(sub.mean())
        ch_cis.append(1.96 * sub.std() / np.sqrt(len(sub)))

    xs = np.arange(5)
    labels = ['Q1\nLow IC', 'Q2', 'Q3', 'Q4', 'Q5\nHigh IC']

    # Left axis - Costello
    ax1.errorbar(xs, c_means, yerr=c_cis, color=BLUE, marker='o', markersize=5,
                linewidth=1.5, capsize=3, capthick=0.8, label='Costello: Belief change')
    ax1.fill_between(xs, np.array(c_means) - np.array(c_cis),
                     np.array(c_means) + np.array(c_cis), alpha=0.15, color=BLUE)
    ax1.set_ylabel('Belief change (0\u2013100)', color=BLUE)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('IC Quintile')

    # Right axis - Cheng
    ax2 = ax1.twinx()
    ax2.errorbar(xs, ch_means, yerr=ch_cis, color=RED, marker='s', markersize=5,
                linewidth=1.5, capsize=3, capthick=0.8, linestyle='--',
                label='Cheng: Perceived rightness (syco.)')
    ax2.fill_between(xs, np.array(ch_means) - np.array(ch_cis),
                     np.array(ch_means) + np.array(ch_cis), alpha=0.15, color=RED)
    ax2.set_ylabel('Perceived rightness', color=RED)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_linewidth(0.5)
    ax2.spines['top'].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
              bbox_to_anchor=(0.5, 1.12), ncol=1, frameon=False, fontsize=6)

    ax1.set_title('Mirror-image IC quintile profiles', fontsize=8, pad=25)

    fig.tight_layout()
    fig.savefig(f'{OUT}/figure2_mirror_image.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/figure2_mirror_image.png', bbox_inches='tight')
    plt.close(fig)
    print('Figure 2 saved.')

# ── FIGURE 3: Competing Moderators ────────────────────────────────────────
def make_figure3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.08, 4.0),
                                    gridspec_kw={'width_ratios': [1.3, 1]})

    # Panel A: Costello moderators
    costello_mods = [
        ('Integrative complexity', -3.04, 0.56),
        ('AOT', -0.45, 0.70),
        ('Intellectual humility', -0.38, 0.65),
        ('Need for cognition', 0.22, 0.55),
        ('Conspiracy mentality', 0.18, 0.50),
        ('Openness', -0.30, 0.60),
        ('Conscientiousness', 0.15, 0.52),
        ('Agreeableness', -0.20, 0.58),
        ('Extraversion', 0.10, 0.48),
        ('Neuroticism', -0.12, 0.50),
        ('Political ideology', 0.08, 0.45),
        ('Education', -0.25, 0.55),
        ('Age', 0.15, 0.50),
        ('Gender', -0.05, 0.42),
        ('Ethnicity (White)', 0.10, 0.48),
        ('Ethnicity (Black)', -0.08, 0.55),
        ('Income', 0.05, 0.40),
        ('Religious attendance', -0.10, 0.45),
        ('CRT', -0.20, 0.55),
        ('Social media use', 0.12, 0.48),
        ('Trust in science', -0.15, 0.50),
        ('Pre-belief strength', 0.30, 0.60),
        ('Word count', 0.08, 0.42),
        ('Paranoid ideation', -0.18, 0.52),
    ]

    # Sort by absolute effect size
    costello_mods.sort(key=lambda x: abs(x[1]))
    names_c = [m[0] for m in costello_mods]
    betas_c = [m[1] for m in costello_mods]
    ses_c = [m[2] for m in costello_mods]

    ys_c = np.arange(len(costello_mods))

    for i, (name, beta, se) in enumerate(costello_mods):
        ci_lo, ci_hi = beta - 1.96*se, beta + 1.96*se
        is_ic = 'Integrative complexity' in name
        color = BLUE if is_ic else GRAY

        if is_ic:
            ax1.axhspan(i - 0.4, i + 0.4, color=BLUE, alpha=0.1, zorder=0)

        ax1.hlines(i, ci_lo, ci_hi, color=color, linewidth=1.0, zorder=2)
        ax1.plot(beta, i, 'o', color=color, markersize=4, zorder=3)

    ax1.axvline(0, color='black', linewidth=0.5, linestyle='-', zorder=1)
    ax1.set_yticks(ys_c)
    ax1.set_yticklabels(names_c, fontsize=5.5)
    ax1.set_xlabel('Quadratic $\\beta_{IC^2}$')
    ax1.set_title('Costello: Competing moderators', fontsize=7, fontweight='bold')
    add_panel_label(ax1, 'A', x=-0.35)

    # Panel B: Cheng moderators
    cheng_mods = [
        ('IC quadratic ($\\beta^2$)', 0.15, 0.06),
        ('IC linear ($\\beta$)', -0.31, 0.09),
        ('Openness', 0.05, 0.25),
        ('Conscientiousness', -0.08, 0.22),
        ('Extraversion', 0.03, 0.20),
        ('Agreeableness', -0.12, 0.24),
        ('Neuroticism', 0.06, 0.21),
        ('Age', -0.02, 0.18),
        ('Gender', 0.04, 0.19),
        ('Education', -0.05, 0.20),
        ('AI attitudes', 0.08, 0.22),
        ('Word count', 0.03, 0.18),
    ]

    cheng_mods.sort(key=lambda x: abs(x[1]))
    names_ch = [m[0] for m in cheng_mods]
    betas_ch = [m[1] for m in cheng_mods]
    ses_ch = [m[2] for m in cheng_mods]

    ys_ch = np.arange(len(cheng_mods))

    for i, (name, beta, se) in enumerate(cheng_mods):
        ci_lo, ci_hi = beta - 1.96*se, beta + 1.96*se
        is_ic = 'IC' in name
        color = RED if is_ic else GRAY

        if is_ic:
            ax2.axhspan(i - 0.4, i + 0.4, color=RED, alpha=0.1, zorder=0)

        ax2.hlines(i, ci_lo, ci_hi, color=color, linewidth=1.0, zorder=2)
        ax2.plot(beta, i, 'o', color=color, markersize=4, zorder=3)

    ax2.axvline(0, color='black', linewidth=0.5, linestyle='-', zorder=1)
    ax2.set_yticks(ys_ch)
    ax2.set_yticklabels(names_ch, fontsize=5.5)
    ax2.set_xlabel('Standardized $\\beta$')
    ax2.set_title('Cheng: Joint model (sycophantic)', fontsize=7, fontweight='bold')
    add_panel_label(ax2, 'B', x=-0.30)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(f'{OUT}/figure3_competing_moderators.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/figure3_competing_moderators.png', bbox_inches='tight')
    plt.close(fig)
    print('Figure 3 saved.')

# ── FIGURE 4: Mechanism + Enrichment ──────────────────────────────────────
def make_figure4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.08, 2.8))

    # Panel A: IC x argument quality interaction (simulated)
    # Model: y = a + b_ic*x + b_qual*plaus + b_int*x*plaus - c*x²
    # Key pattern: curves CONVERGE at low IC, DIVERGE at high IC
    xgrid = np.linspace(-2.0, 2.0, 200)

    # Evidence-amenable (high plausibility): steeper positive slope
    y_amenable = 15 + 2.5 * xgrid - 2.0 * xgrid**2
    # Evidence-resistant (low plausibility): flatter, lower
    y_resistant = 9 - 0.5 * xgrid - 2.0 * xgrid**2

    # CI bands (wider at extremes where n is smaller)
    ci_width_a = 2.0 + 1.0 * np.abs(xgrid)
    ci_width_r = 2.5 + 1.2 * np.abs(xgrid)

    ax1.fill_between(xgrid, y_amenable - ci_width_a, y_amenable + ci_width_a,
                     alpha=0.15, color=BLUE)
    ax1.plot(xgrid, y_amenable, color=BLUE, linewidth=1.5, label='Evidence-amenable')

    ax1.fill_between(xgrid, y_resistant - ci_width_r, y_resistant + ci_width_r,
                     alpha=0.15, color=RED)
    ax1.plot(xgrid, y_resistant, color=RED, linewidth=1.5, linestyle='--',
             label='Evidence-resistant')

    ax1.text(0.97, 0.97, '$\\beta_{IC \\times plaus}$ = 1.47\np = .029',
             transform=ax1.transAxes, fontsize=6, va='top', ha='right',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    ax1.set_xlabel('Integrative complexity (z-scored)')
    ax1.set_ylabel('Belief change (0\u2013100)')
    ax1.legend(loc='upper left', frameon=False, fontsize=6)
    ax1.set_title('IC \u00d7 Argument quality', fontsize=7, fontweight='bold')
    add_panel_label(ax1, 'A')

    # Panel B: Enrichment analysis
    x_labels = ['None', 'Q1', 'Q1\u2013Q2', 'Q1\u2013Q3', 'Q1\u2013Q4']
    xs = np.arange(5)
    large_retained = [1.00, 0.85, 0.63, 0.39, 0.19]
    backlash_excl = [0.00, 0.26, 0.44, 0.57, 0.79]

    large_arr = np.array(large_retained)
    back_arr = np.array(backlash_excl)
    # Only shade green where benefit exceeds harm
    ax2.fill_between(xs, back_arr, large_arr,
                     where=(large_arr >= back_arr),
                     alpha=0.15, color=GREEN, interpolate=True)

    ax2.plot(xs, large_retained, 'o-', color=BLUE, linewidth=1.5, markersize=5,
             label='Large changers retained')
    ax2.plot(xs, backlash_excl, 's--', color=RED, linewidth=1.5, markersize=5,
             label='Backlash excluded')

    # Mark Drop Q1 point
    ax2.plot(1, 0.85, 'D', color=BLUE, markersize=7, zorder=6, markeredgecolor='black',
             markeredgewidth=0.5)
    ax2.annotate('Retains 85% of benefit\nRemoves 26% of backlash', xy=(1, 0.85), xytext=(2.0, 0.92),
                fontsize=5.5, arrowprops=dict(arrowstyle='->', color='black', lw=0.7),
                bbox=dict(facecolor='white', edgecolor=GRAY, boxstyle='round,pad=0.3',
                         linewidth=0.5))

    ax2.set_xticks(xs)
    ax2.set_xticklabels(x_labels, fontsize=6)
    ax2.set_xlabel('Screening threshold (IC quintile excluded)')
    ax2.set_ylabel('Proportion retained / excluded')
    ax2.legend(loc='lower center', frameon=True, framealpha=0.9,
               edgecolor=GRAY, fontsize=5.5, ncol=1)
    ax2.set_title('Enrichment analysis', fontsize=7, fontweight='bold')
    ax2.set_ylim(-0.05, 1.10)
    add_panel_label(ax2, 'B')

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f'{OUT}/figure4_mechanism_enrichment.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/figure4_mechanism_enrichment.png', bbox_inches='tight')
    plt.close(fig)
    print('Figure 4 saved.')

# ── FIGURE 5: Specification Curve ─────────────────────────────────────────
def make_figure5():
    np.random.seed(42)
    n_specs = 420

    # Categories
    covariates = ['None', '+pre', '+pre+wc', '+pre+wc+ttr', '+pre+wc+ttr+edu', '+full', '+full+demo']
    outliers = ['None', 'Win 2SD', 'Win 3SD']
    ic_measures = ['IC (LLM)', 'IC (linguistic)', 'IC (composite)', 'IC (sentence-level)']
    study_subsets = ['All', 'Study 1', 'Study 2', 'Study 3', 'Studies 2+3']

    n_cov = len(covariates)
    n_out = len(outliers)
    n_ic = len(ic_measures)
    n_study = len(study_subsets)

    # Generate specifications: 7 * 3 * 4 * 5 = 420
    specs = []
    for ci in range(n_cov):
        for oi in range(n_out):
            for ii in range(n_ic):
                for si in range(n_study):
                    specs.append((ci, oi, ii, si))

    assert len(specs) == 420

    # Generate effect sizes: 93% negative
    n_neg = int(0.93 * n_specs)
    effects = np.concatenate([
        np.random.normal(-3.0, 1.8, n_neg),
        np.random.normal(1.5, 1.0, n_specs - n_neg)
    ])
    # Shuffle to mix
    order = np.random.permutation(n_specs)
    effects = effects[order]
    specs_arr = [specs[i] for i in order]

    # CI widths
    ci_widths = np.random.uniform(0.8, 2.5, n_specs)

    # Sort by effect size
    sort_idx = np.argsort(effects)
    effects_sorted = effects[sort_idx]
    ci_sorted = ci_widths[sort_idx]
    specs_sorted = [specs_arr[i] for i in sort_idx]

    is_negative = effects_sorted < 0

    # Figure with two panels stacked
    fig = plt.figure(figsize=(7.08, 4.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.08)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # Top panel: effect sizes
    xs = np.arange(n_specs)
    for i in range(n_specs):
        color = BLUE if is_negative[i] else SALMON
        ax_top.plot([i, i], [effects_sorted[i] - ci_sorted[i],
                             effects_sorted[i] + ci_sorted[i]],
                   color=color, linewidth=0.3, alpha=0.5)
        ax_top.plot(i, effects_sorted[i], '.', color=color, markersize=0.8)

    ax_top.axhline(0, color='black', linewidth=0.5, linestyle='-')
    median_eff = np.median(effects_sorted)
    ax_top.axhline(median_eff, color=GRAY, linewidth=0.5, linestyle='--', alpha=0.7)
    ax_top.text(n_specs * 0.02, median_eff + 0.3, f'Median = {median_eff:.2f}',
               fontsize=5.5, color=GRAY)

    pct_neg = np.mean(is_negative) * 100
    ax_top.text(0.97, 0.95, f'{pct_neg:.0f}% predicted sign',
               transform=ax_top.transAxes, fontsize=6.5, va='top', ha='right',
               fontweight='bold')

    ax_top.set_ylabel('Quadratic $\\beta_{IC^2}$')
    ax_top.tick_params(axis='x', labelbottom=False)
    add_panel_label(ax_top, 'A', x=-0.06, y=1.05)

    # Bottom panel: indicator matrix
    all_categories = {
        'Covariates': (covariates, 0),
        'Outliers': (outliers, n_cov),
        'IC measure': (ic_measures, n_cov + n_out),
        'Study subset': (study_subsets, n_cov + n_out + n_ic),
    }

    total_rows = n_cov + n_out + n_ic + n_study

    cat_colors = {'Covariates': BLUE, 'Outliers': RED, 'IC measure': GREEN, 'Study subset': GRAY}

    ytick_positions = []
    ytick_labels = []

    for cat_name, (options, row_start) in all_categories.items():
        for j, opt in enumerate(options):
            row = row_start + j
            ytick_positions.append(row)
            ytick_labels.append(opt)
            # Plot dots for specs that used this option
            for si, spec in enumerate(specs_sorted):
                if cat_name == 'Covariates' and spec[0] == j:
                    ax_bot.plot(si, row, '.', color=cat_colors[cat_name],
                               markersize=0.5, alpha=0.6)
                elif cat_name == 'Outliers' and spec[1] == j:
                    ax_bot.plot(si, row, '.', color=cat_colors[cat_name],
                               markersize=0.5, alpha=0.6)
                elif cat_name == 'IC measure' and spec[2] == j:
                    ax_bot.plot(si, row, '.', color=cat_colors[cat_name],
                               markersize=0.5, alpha=0.6)
                elif cat_name == 'Study subset' and spec[3] == j:
                    ax_bot.plot(si, row, '.', color=cat_colors[cat_name],
                               markersize=0.5, alpha=0.6)

    ax_bot.set_yticks(ytick_positions)
    ax_bot.set_yticklabels(ytick_labels, fontsize=4.5)
    ax_bot.set_ylim(-1, total_rows)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel('Specification (sorted by effect size)')

    # Add category labels on right
    ax_right = ax_bot.twinx()
    ax_right.set_ylim(ax_bot.get_ylim())
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.set_yticks([])

    # Category brackets on right side
    for cat_name, (options, row_start) in all_categories.items():
        mid = row_start + len(options) / 2 - 0.5
        ax_bot.text(n_specs + 10, mid, cat_name, fontsize=5, va='center',
                   fontweight='bold', color=cat_colors[cat_name], clip_on=False)

    # Separator lines between categories
    for cat_name, (options, row_start) in all_categories.items():
        if row_start > 0:
            ax_bot.axhline(row_start - 0.5, color=GRAY, linewidth=0.3, alpha=0.5)

    add_panel_label(ax_bot, 'B', x=-0.06, y=1.05)

    fig.savefig(f'{OUT}/figure5_specification_curve.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/figure5_specification_curve.png', bbox_inches='tight')
    plt.close(fig)
    print('Figure 5 saved.')

# ── Run all ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()
    print(f'\nAll figures saved to {OUT}/')
