#!/usr/bin/env python3
"""verify_si_zic2_rescale.py — INDEPENDENT re-derivation of SI table values on the
headline z(IC)^2 scale (z-score THEN square), from raw data only.

Convention (from task spec): "z(IC)^2 scale" = z-score the predictor over the
analysis sample, THEN square. Headline model: DV ~ z(M) + z(M)^2 + z(pre) + z(wc).
Anchor: beta_IC2 = -1.99 on Costello primary Q400 logit-EV.

NOTE: the GOLD scripts use zs(x**2) = z(IC^2) (square-then-z, the OLD scale).
This harness uses zs(x)**2 = z(IC)^2 (the headline scale) throughout. Conversion
factor differs per predictor: SD(IC^2)/SD(IC)^2 (re-derived each time).

REPORT-ONLY. Does not edit any existing file.
"""
import json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DCOST = ROOT / "data" / "costello2024"
DQ400 = ROOT / "data" / "ic_qwen3orpo400"


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def bf10_bic(y, X_red, X_full):
    """BIC-approx BF10 for full vs reduced (positive => favors full)."""
    m0 = sm.OLS(y, X_red).fit()
    m1 = sm.OLS(y, X_full).fit()
    n = len(y)
    bic0 = n * np.log(m0.ssr / n) + X_red.shape[1] * np.log(n)
    bic1 = n * np.log(m1.ssr / n) + X_full.shape[1] * np.log(n)
    return np.exp((bic0 - bic1) / 2.0)


def quad_zic2(y, m_raw, cov_list_z):
    """Fit DV ~ z(M) + z(M)^2 + covs (covs already z-scored). z(IC)^2 scale.
    Returns dict with beta_quad, p, BF (quad vs linear), apex (raw-IC), n."""
    zm = zs(m_raw)
    zm2 = zm ** 2          # <-- z THEN square (headline scale)
    X_lin = sm.add_constant(np.column_stack([zm] + cov_list_z))
    X_quad = sm.add_constant(np.column_stack([zm, zm2] + cov_list_z))
    mq = sm.OLS(y, X_quad).fit()
    bf = bf10_bic(y, X_lin, X_quad)
    # apex on raw-IC scale: dDV/dIC=0 => need beta on z scale -> convert
    b1, b2 = mq.params[1], mq.params[2]
    mu, sd = np.nanmean(m_raw), np.nanstd(m_raw)
    # vertex in z units: z* = -b1/(2 b2); raw apex = mu + sd*z*
    zstar = -b1 / (2 * b2)
    apex_raw = mu + sd * zstar
    return dict(beta_lin=b1, beta_quad=b2, p_quad=mq.pvalues[2], bf=bf,
                apex=apex_raw, n=len(y), r2=mq.rsquared)


# ---------------- Costello loaders ----------------
def load_costello_primary():
    an = pd.read_csv(DCOST / "analysis_data.csv")
    ic = pd.read_csv(DQ400 / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    ic = ic[["participantId", "ic_qwenorpo400_logit"]].rename(
        columns={"ic_qwenorpo400_logit": "IC"})
    df = an.merge(ic, on="participantId", how="inner")
    for c in ["DV_BeliefChange_Specific", "Pre_Belief_Specific",
              "OpenendedResponseWordCount", "IC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


SEP = "=" * 78


def report(label, sival, mine, tol=0.03, note=""):
    """tol is absolute on the coefficient scale."""
    try:
        match = abs(float(sival) - float(mine)) <= tol
    except Exception:
        match = None
    tag = "MATCH" if match else ("MISMATCH" if match is False else "?")
    print(f"[{tag:8s}] {label:48s} si={sival:>8} mine={mine:>9} {note}")


def main():
    df = load_costello_primary()
    dv = "DV_BeliefChange_Specific"
    pre = "Pre_Belief_Specific"
    wc = "OpenendedResponseWordCount"

    # ===== ANCHOR =====
    print(SEP); print("ANCHOR: Costello headline beta_IC2 (z(IC)^2)"); print(SEP)
    sub = df.dropna(subset=[dv, "IC", pre, wc]).copy()
    y = sub[dv].values
    r = quad_zic2(y, sub["IC"].values,
                  [zs(sub[pre].values), zs(sub[wc].values)])
    report("Headline beta_IC2", "-1.99", round(r["beta_quad"], 3),
           note=f"n={r['n']} apex={r['apex']:.2f} BF={r['bf']:.0f}")

    # ===== ITEM 4: tab:costello_movers =====
    print(SEP); print("ITEM 4: tab:costello_movers (|Delta| thresholds)"); print(SEP)
    thr_si = {0: -1.80, 1: -1.99, 2: -2.12, 5: -2.48, 10: -2.55}
    sub_all = df.dropna(subset=[dv, "IC", pre, wc]).copy()
    print(f" full n={len(sub_all)}")
    for t, sv in thr_si.items():
        s = sub_all[np.abs(sub_all[dv]) > t].copy()
        rr = quad_zic2(s[dv].values, s["IC"].values,
                       [zs(s[pre].values), zs(s[wc].values)])
        report(f"|D|>{t}", f"{sv}", round(rr["beta_quad"], 2),
               note=f"n={rr['n']} kept={100*rr['n']/len(sub_all):.1f}% "
                    f"apex={rr['apex']:.2f} BF={rr['bf']:.1f} p={rr['p_quad']:.4f}")

    # ===== ITEM 2: tab:incremental_validity =====
    print(SEP); print("ITEM 2: tab:incremental_validity (demographic adj)"); print(SEP)
    # need demographic cols from analysis_data + raw
    raw_path = DCOST / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"
    extra = None
    if raw_path.exists():
        raw = pd.read_csv(raw_path, low_memory=False).drop_duplicates(
            subset="participantId", keep="first")
    # Use analysis_data demographic-adjacent columns
    print(" analysis_data has:", [c for c in
          ["AgeYears", "Education_Cat", "Extremism", "religion", "genai_trust"]
          if c in df.columns])
    dfx = df.copy()
    # education numeric
    edu_map = {"LessThanHighSchool": 1, "HighSchool": 2, "SomeCollege": 3,
               "Associate": 3, "Bachelors": 4, "Masters": 5, "JD/MD": 6, "PhD": 6}
    if "Education_Cat" in dfx.columns:
        dfx["edu_num"] = dfx["Education_Cat"].map(edu_map)
    for c in ["AgeYears", "Extremism", "religion", "genai_trust"]:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    # TTR from surface features
    surf = pd.read_csv(DCOST / "costello_surface_features.csv")
    if "surf_ttr" in surf.columns:
        dfx = dfx.merge(surf[["participantId", "surf_ttr"]], on="participantId",
                        how="left")

    base = dfx.dropna(subset=[dv, "IC", pre, wc]).copy()
    # canonical full sample
    rc = quad_zic2(base[dv].values, base["IC"].values,
                   [zs(base[pre].values), zs(base[wc].values)])
    report("Canonical full sample", "-1.99", round(rc["beta_quad"], 2),
           note=f"n={rc['n']} BF={rc['bf']:.0f}")

    # 5-cov main-text headline: Age + Edu + Extremism + Religion + GenAI-trust
    cov5 = ["AgeYears", "edu_num", "Extremism", "religion", "genai_trust"]
    cov5 = [c for c in cov5 if c in dfx.columns]
    s5 = dfx.dropna(subset=[dv, "IC", pre, wc] + cov5).copy()
    cz5 = [zs(s5[pre].values), zs(s5[wc].values)] + [zs(s5[c].values) for c in cov5]
    r5 = quad_zic2(s5[dv].values, s5["IC"].values, cz5)
    report("5-cov (Age+Edu+Extrem+Relig+GenAI)", "-1.91",
           round(r5["beta_quad"], 2),
           note=f"n={r5['n']} covs={cov5} apex={r5['apex']:.2f}")

    # composition-only subsample (same N as full-covariate, canonical model)
    rcomp = quad_zic2(s5[dv].values, s5["IC"].values,
                      [zs(s5[pre].values), zs(s5[wc].values)])
    report("Composition-only subsample (canon)", "-1.97",
           round(rcomp["beta_quad"], 2), note=f"n={rcomp['n']}")

    # verbal-fluency variant: Age + Edu + Extremism + TTR
    covV = ["AgeYears", "edu_num", "Extremism", "surf_ttr"]
    covV = [c for c in covV if c in dfx.columns]
    sV = dfx.dropna(subset=[dv, "IC", pre, wc] + covV).copy()
    czV = [zs(sV[pre].values), zs(sV[wc].values)] + [zs(sV[c].values) for c in covV]
    rV = quad_zic2(sV[dv].values, sV["IC"].values, czV)
    report("Verbal-fluency variant (Age+Edu+Ext+TTR)", "-1.95",
           round(rV["beta_quad"], 2), note=f"n={rV['n']} covs={covV}")

    # retention 97.9%
    if r5["n"] == rcomp["n"]:
        ret = r5["beta_quad"] / rcomp["beta_quad"] * 100
        print(f"  retention (5cov vs composition-only) = {ret:.1f}% "
              f"(si says 97.9%)")
    ret_full = r5["beta_quad"] / rc["beta_quad"] * 100
    print(f"  5cov / canonical-full = {ret_full:.1f}% (96.0% in si text)")

    # ===== ITEM 3: tab:aiic_survival =====
    print(SEP); print("ITEM 3: tab:aiic_survival (AI-side controls)"); print(SEP)
    gpt = pd.read_csv(DCOST / "costello_gpt_ic_qwenorpo400.csv")
    gcols = ["participantId", "ic_r1_logit", "ic_r2_logit", "ic_r3_logit",
             "ic_concat_logit"]
    # gpt file is long-format with duplicate participantIds; dedup to 1 row/pid
    g = gpt[gcols].drop_duplicates("participantId").copy()
    g["gpt_mean"] = g[["ic_r1_logit", "ic_r2_logit", "ic_r3_logit"]].mean(axis=1)
    g["gpt_slope"] = g["ic_r3_logit"] - g["ic_r1_logit"]
    g["gpt_sd"] = g[["ic_r1_logit", "ic_r2_logit", "ic_r3_logit"]].std(axis=1)
    dg = df.merge(g, on="participantId", how="left")
    bb = dg.dropna(subset=[dv, "IC", pre, wc]).copy()
    # reference
    rref = quad_zic2(bb[dv].values, bb["IC"].values,
                     [zs(bb[pre].values), zs(bb[wc].values)])
    report("(none) reference", "-1.99", round(rref["beta_quad"], 2),
           note=f"n={rref['n']}")
    for extra_cols, sv, name in [
            (["gpt_mean"], -2.00, "+gpt_mean"),
            (["ic_concat_logit"], -1.98, "+gpt_concat"),
            (["gpt_mean", "gpt_slope", "gpt_sd"], -1.88, "+mean,slope,SD")]:
        s = dg.dropna(subset=[dv, "IC", pre, wc] + extra_cols).copy()
        cz = [zs(s[pre].values), zs(s[wc].values)] + \
             [zs(s[c].values) for c in extra_cols]
        rr = quad_zic2(s[dv].values, s["IC"].values, cz)
        report(name, f"{sv}", round(rr["beta_quad"], 2),
               note=f"n={rr['n']} R2={rr['r2']:.3f}")

    # ===== ITEM 5: tab:surface_ablation =====
    print(SEP); print("ITEM 5: tab:surface_ablation"); print(SEP)
    feats = ["surf_wc", "surf_fk", "surf_dc", "surf_smog", "surf_asl",
             "surf_ttr", "surf_marker"]
    si_feat = {"surf_wc": -0.22, "surf_fk": -0.04, "surf_dc": -0.57,
               "surf_smog": -0.43, "surf_asl": -0.24, "surf_ttr": -0.71,
               "surf_marker": -0.69}
    dsf = df.merge(surf, on="participantId", how="left")
    # NB: standalone rows are computed on the COMMON complete-case sample
    # (all surf feats + IC), matching note20; per-feature dropna gives wrong N.
    common = dsf.dropna(subset=[dv, "IC", pre, wc] + feats).copy()
    for f in feats:
        s = common
        rr = quad_zic2(s[dv].values, s[f].values,
                       [zs(s[pre].values), zs(s[wc].values)])
        report(f"{f} beta_quad", f"{si_feat[f]}", round(rr["beta_quad"], 2),
               note=f"n={rr['n']} p={rr['p_quad']:.3f} BF={rr['bf']:.2f}")
    # omnibus: IC^2 with all 14 surface terms as covs
    so = dsf.dropna(subset=[dv, "IC", pre, wc] + feats).copy()
    covz = [zs(so[pre].values), zs(so[wc].values)]
    for f in feats:
        covz.append(zs(so[f].values))
        covz.append(zs(so[f].values) ** 2)   # z(feat)^2 surface quad controls
    zic = zs(so["IC"].values)
    zic2 = zic ** 2
    X_lin = sm.add_constant(np.column_stack([zic] + covz))
    X_full = sm.add_constant(np.column_stack([zic, zic2] + covz))
    mq = sm.OLS(so[dv].values, X_full).fit()
    bf = bf10_bic(so[dv].values, X_lin, X_full)
    report("omnibus IC^2 (14 surface controls)", "-1.71",
           round(mq.params[2], 2),
           note=f"n={len(so)} p={mq.pvalues[2]:.4f} BF={bf:.1f}")

    # ===== ITEM 6: tab:costello_engagement =====
    print(SEP); print("ITEM 6: tab:costello_engagement (interaction BFs)"); print(SEP)
    # Moderators: topic importance, pre-belief strength, word count
    # ICxM and IC^2 x M; joint BF against 2-interaction alternative.
    # NB: Importance has a -999 missing sentinel that MUST be removed (gives
    # N=1210, not 1286). Controls differ per moderator (the moderator itself
    # is excluded from its own control set).
    dfe = df.copy()
    if "Importance" in dfe.columns:
        dfe["Importance"] = pd.to_numeric(dfe["Importance"], errors="coerce")
        dfe.loc[dfe["Importance"] <= -90, "Importance"] = np.nan

    def engagement_row(modcol, ctrl_cols, subdf):
        s = subdf.dropna(subset=[dv, "IC", modcol] + ctrl_cols).copy()
        y = s[dv].values
        zic = zs(s["IC"].values); zic2 = zic ** 2
        zm = zs(s[modcol].values)
        ctrls = [zs(s[c].values) for c in ctrl_cols]
        base_cols = [zic, zic2, zm] + ctrls
        X_base = sm.add_constant(np.column_stack(base_cols))
        X_1 = sm.add_constant(np.column_stack([zic, zic2, zm, zic * zm] + ctrls))
        X_2 = sm.add_constant(np.column_stack(
            [zic, zic2, zm, zic * zm, zic2 * zm] + ctrls))
        m1 = sm.OLS(y, X_1).fit()
        m2 = sm.OLS(y, X_2).fit()
        bf_first = bf10_bic(y, X_base, X_1)   # ICxM over base
        bf_second = bf10_bic(y, X_1, X_2)     # IC^2xM over (base+ICxM)
        bf_joint = bf10_bic(y, X_base, X_2)   # both over base
        return dict(n=len(s),
                    b_first=m1.params[3], p_first=m1.pvalues[3], bf_first=bf_first,
                    b_second=m2.params[4], p_second=m2.pvalues[4],
                    bf_second=bf_second, bf_joint=bf_joint)

    rows6 = []
    if "Importance" in dfe.columns:
        rows6.append(("Topic importance",
                      engagement_row("Importance", [pre, wc], dfe)))
    rows6.append(("Pre-belief strength", engagement_row(pre, [wc], dfe)))
    rows6.append(("Word count", engagement_row(wc, [pre], dfe)))
    for nm, rr in rows6:
        print(f"  {nm:20s} n={rr['n']:4d}  ICxM b={rr['b_first']:+.2f} "
              f"p={rr['p_first']:.3f} BF={rr['bf_first']:.3f} | "
              f"IC2xM b={rr['b_second']:+.2f} p={rr['p_second']:.3f} "
              f"BF={rr['bf_second']:.3f} | jointBF={rr['bf_joint']:.4f}")
    maxsecond = max(r["bf_second"] for _, r in rows6)
    maxjoint = max(r["bf_joint"] for _, r in rows6)
    maxfirst = max(r["bf_first"] for _, r in rows6)
    print(f"  MAX first-order BF = {maxfirst:.3f} (si: <=0.04)")
    print(f"  MAX second-order BF = {maxsecond:.3f} (si: <=0.17; main text <=0.15)")
    print(f"  MAX joint BF = {maxjoint:.4f} (si: <=0.006; main text <=0.005)")

    # ===== ITEM 1: tab:alt_constructs (re-scored from raw EV columns) =====
    print(SEP); print("ITEM 1: tab:alt_constructs (z(C)^2, gpt-mini IC ref)"); print(SEP)
    alt = pd.read_csv(DCOST / "alt_constructs_logitev.csv")[
        ["participantId", "AOT_ev", "IH_ev", "NFC_ev", "OMI_ev"]]
    icg = pd.read_csv(DCOST / "all_complexity_scores.csv")[
        ["participantId", "IC_openai"]].rename(columns={"IC_openai": "ICg"})
    da = df.merge(icg, on="participantId", how="inner").merge(
        alt, on="participantId", how="left")
    for c in ["ICg", "AOT_ev", "IH_ev", "NFC_ev", "OMI_ev"]:
        da[c] = pd.to_numeric(da[c], errors="coerce")
    si_alt = {"ICg": -3.04, "AOT_ev": -2.36, "IH_ev": -2.06,
              "NFC_ev": -4.70, "OMI_ev": -1.91}
    for col, sv in si_alt.items():
        s = da.dropna(subset=[dv, col, pre, wc])
        rr = quad_zic2(s[dv].values, s[col].values,
                       [zs(s[pre].values), zs(s[wc].values)])
        report(f"{col}", f"{sv}", round(rr["beta_quad"], 2),
               note=f"n={rr['n']} apex={rr['apex']:.2f} BF={rr['bf']:.3g}")

    # ===== ITEM 8: tab:discriminant (incremental-F, scale-invariant) =====
    print(SEP); print("ITEM 8: tab:discriminant Costello row (2-df inc-F)"); print(SEP)
    s = df.dropna(subset=[dv, "IC", pre, wc])
    y = s[dv].values
    Xc = sm.add_constant(np.column_stack([zs(s[wc].values), zs(s[pre].values)]))
    zic = zs(s["IC"].values)
    Xf = sm.add_constant(np.column_stack(
        [zs(s[wc].values), zs(s[pre].values), zic, zic ** 2]))
    m0 = sm.OLS(y, Xc).fit(); m1 = sm.OLS(y, Xf).fit()
    dr2 = m1.rsquared - m0.rsquared
    F = (dr2 / 2) / ((1 - m1.rsquared) / m1.df_resid)
    p = sp.f.sf(F, 2, m1.df_resid)
    report("Costello dR2", ".013", round(dr2, 3), tol=0.001,
           note=f"F(2,{int(m1.df_resid)})={F:.1f} p={p:.4g} (si F=11.7)")
    print("  (Boissin/Cheng rows: see note_discriminant_table.py;"
          " all scale-invariant 2-df inc-F)")

    # ===== ITEM 9: Note 25 Likert power-loss -1.68 =====
    print(SEP); print("ITEM 9: Note 25 Likert subsample -1.68"); print(SEP)
    for c in ["IH", "AOT", "StudyNumber"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # full Study 1
    s1 = df[df["StudyNumber"] == 1].dropna(subset=[dv, "IC", pre, wc])
    r1 = quad_zic2(s1[dv].values, s1["IC"].values,
                   [zs(s1[pre].values), zs(s1[wc].values)])
    print(f"  Study-1 ALL: n={r1['n']} beta_IC2={r1['beta_quad']:+.3f} "
          f"p={r1['p_quad']:.3f}")
    # Likert subsample = Study 1 complete-case on IH & AOT (N=325)
    sl = df[(df["StudyNumber"] == 1) & df["IH"].notna() &
            df["AOT"].notna()].dropna(subset=[dv, "IC", pre, wc])
    rl = quad_zic2(sl[dv].values, sl["IC"].values,
                   [zs(sl[pre].values), zs(sl[wc].values)])
    report("Likert subsample (Study1+IH&AOT, N=325)", "-1.68",
           round(rl["beta_quad"], 2),
           note=f"n={rl['n']} p={rl['p_quad']:.3f} BF={rl['bf']:.3f}")


if __name__ == "__main__":
    main()
