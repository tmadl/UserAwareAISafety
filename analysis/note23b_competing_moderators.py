#!/usr/bin/env python3
"""note23b_competing_moderators.py — canonical, reproducible rebuild of the
24-moderator head-to-head comparison (manuscript Table tab:moderators and the
"IC is the strongest moderator, >2.5x misinformation resistance" claim, main
"IC outperforms competing moderators").

Background: the original table was produced by scripts/14_competing_moderators.py
(main repo, NOT in the release), on the z(IC^2) parameterisation and the minimal
"pre-belief only" covariate spec. After the headline switch to z(IC)^2, that
table became (a) scale-mixed (the q400 IC row was converted to z(IC)^2 = -1.97
but the gpt/Conway rows were left on raw IC^2), and (b) non-reproducible from the
release. This script rebuilds it from release data on a single z(IC)^2 scale,
under BOTH covariate specs, so the manuscript can report reproduced numbers.

Each continuous moderator M:
  quad : DV ~ z(M) + z(M)^2 + covs        (b_quad on z(M)^2 -- headline scale)
  lin  : DV ~ z(M)       + covs            (b_lin = linear-only coefficient)
  dR2  : R^2(quad) - R^2(lin)              (the quadratic moderation's contribution)
Binary moderators (gender, party): linear only.

Two covariate specs reported side by side:
  SPEC A (minimal, what 14 used):  covs = z(pre-belief)
  SPEC B (headline):               covs = z(pre-belief) + z(word count)

Validation anchors (must hold or the harness is wrong):
  q400 IC  b_quad ~ -1.97 (z(IC)^2);  gpt-4.1-mini ~ -2.08;  Conway ~ +0.18.

Output: prints a per-spec table + IC-vs-misinformation-resistance dR2 ratio;
writes note23b_competing_moderators_output.txt. No manuscript edits.
"""
import importlib.util
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
REL = HERE.parent
DCOST = REL / "data" / "costello2024"
DQ400 = REL / "data" / "ic_qwen3orpo400"
RAW = DCOST / "Data 8.28.24" / "AllDataForPublication.PPI.8.28.24.csv"


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / np.nanstd(v, ddof=1)


def ttr(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan
    toks = text.lower().split()
    return len(set(toks)) / max(len(toks), 1)


def load():
    raw = pd.read_csv(RAW, low_memory=False)
    df = raw.drop_duplicates(subset=["participantId"], keep="first").copy()
    df = df[df["ExperimentalCondition"] == "Treatment"].copy()

    num = ["DV_BeliefChange_Specific", "Pre_Belief_Specific", "OpenendedResponseWordCount",
           "Sureness_1", "Importance", "Misinformation_Resis", "Social_Influence2",
           "GeneralTrust", "PersonalTrust", "InstitutionalTrust", "genai_trust",
           "genai_fam_1", "genai_use_1", "Extremism", "religion", "IH", "AOT"]
    # The -999 "refused/missing" survey sentinel applies only to the survey
    # moderators. DV/pre/word count have legitimate ranges (DV in [-100, 100]),
    # so the sentinel filter must NOT touch them -- otherwise large adverse
    # movers (DV < -90) are silently nulled, biasing every moderator's quadratic
    # (it shifted the IC row from the headline -1.99 to -2.07).
    no_sentinel = {"DV_BeliefChange_Specific", "Pre_Belief_Specific",
                   "OpenendedResponseWordCount"}
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if c not in no_sentinel:
                df.loc[df[c] < -90, c] = np.nan
    # age may be under a different name
    for cand in ["age", "Age", "age_1", "Age_1"]:
        if cand in df.columns:
            df["age"] = pd.to_numeric(df[cand], errors="coerce")
            break

    edu_map = {"LessThanHighSchool": 1, "HighSchool": 2, "SomeCollege": 3, "Associate": 3,
               "Bachelors": 4, "Masters": 5, "JD/MD": 6, "PhD": 6}
    if "Education_Cat" in df.columns:
        df["edu_num"] = df["Education_Cat"].map(edu_map)
    if "GenderCat" in df.columns:
        df["is_male"] = (df["GenderCat"] == "Male").astype(float)
    if "PartyCat" in df.columns:
        df["is_republican"] = df["PartyCat"].str.contains("Repub", na=False).astype(float)
        df["is_democrat"] = df["PartyCat"].str.contains("Dem", na=False).astype(float)
    if "userResponse_combined" in df.columns:
        df["ttr"] = df["userResponse_combined"].apply(ttr)
        df["initial_wc"] = df["userResponse_combined"].apply(
            lambda x: len(x.split()) if isinstance(x, str) else np.nan)
    for c in ["ParticipantResponseWordCount_r1", "ParticipantResponseWordCount_r2",
              "ParticipantResponseWordCount_r3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    wcs = [c for c in ["ParticipantResponseWordCount_r1", "ParticipantResponseWordCount_r2",
                       "ParticipantResponseWordCount_r3"] if c in df.columns]
    if wcs:
        df["total_participant_wc"] = df[wcs].sum(axis=1)

    # IC scorers
    meta = [json.loads(l) for l in open(DCOST / "texts_for_scoring.jsonl")]
    q = pd.read_csv(DQ400 / "costello_texts_for_scoring_initial_qwenorpo400.csv")
    q400 = pd.DataFrame({"participantId": [m["participantId"] for m in meta],
                         "IC_q400": q["ic_qwenorpo400_logit"].astype(float).values})
    gpt = pd.read_csv(DCOST / "all_complexity_scores.csv")[
        ["participantId", "IC_openai_initial"]].rename(columns={"IC_openai_initial": "IC_gpt"})
    conway = pd.read_csv(DCOST / "costello_conway_ic.csv").rename(columns={"ic_conway": "IC_conway"})
    # AI-side IC (GPT mean over the three reply turns, logit-EV); see note19.
    gic = pd.read_csv(DCOST / "costello_gpt_ic_qwenorpo400.csv")
    gic["IC_aiside"] = gic[["ic_r1_logit", "ic_r2_logit", "ic_r3_logit"]].mean(axis=1)
    aiside = gic[["participantId", "IC_aiside"]]
    for t in (q400, gpt, conway, aiside):
        df = df.merge(t.drop_duplicates(subset="participantId"), on="participantId", how="left")
    # a merge table carried duplicate participantIds, double-counting 3
    # participants (1782 -> 1785) and biasing IC to -2.02; dedup to be safe.
    df = df.drop_duplicates(subset="participantId", keep="first")
    # canonical headline sample (analysis_data.csv) -- restrict the complexity
    # family to it so the IC row reproduces the headline beta exactly (-1.99),
    # rather than RAW dedup's slightly larger set.
    canon = set(pd.read_csv(DCOST / "analysis_data.csv")["participantId"])
    df["_canon"] = df["participantId"].isin(canon)
    return df


# Aligned to manuscript Table tab:moderators (row order + labels).
MODS = [
    # (col, label, category, binary?)
    ("IC_q400", "IC", "Complexity (user)", False),
    ("IC_gpt", "IC (gpt-4.1-mini baseline)", "Complexity (user)", False),
    ("IC_conway", "IC (Conway AutoIC)", "Complexity (user)", False),
    ("IC_aiside", "AI-side IC (GPT mean, logit-EV)", "Complexity (AI)", False),
    ("Sureness_1", "Belief sureness", "Belief", False),
    ("Importance", "Topic importance", "Belief", False),
    ("age", "Age", "Demographic", False),
    ("edu_num", "Education", "Demographic", False),
    ("is_male", "Gender (male)", "Demographic", True),
    ("religion", "Religiosity", "Demographic", False),
    ("Extremism", "Pol. extremism", "Demographic", False),
    ("IH", "Intellectual humility", "Personality", False),
    ("AOT", "Open-minded thinking", "Personality", False),
    ("Social_Influence2", "Social influence", "Personality", False),
    ("Misinformation_Resis", "Misinfo resistance", "Personality", False),
    ("InstitutionalTrust", "Institutional trust", "Trust", False),
    ("PersonalTrust", "Personal trust", "Trust", False),
    ("genai_trust", "Trust in AI", "Trust", False),
    ("initial_wc", "Word count", "Surface", False),
]


def fit_mod(df, col, binary, covset, common_cols=None, canon_only=False):
    """Return (n, b_lin, p_lin, b_quad, p_quad, dr2, collin) under covset ('A'/'B').

    `collin` is the R^2 of regressing the (z-scored) moderator on the z-scored
    covariates; values near 1.0 mean the moderator is (near-)perfectly explained
    by pre-belief / word count, so its LINEAR coefficient is not identified and
    blows up numerically. The quadratic term remains identified (z(M)^2 adds a
    non-collinear curvature dimension) and is still reported.

    `common_cols`: if given, the row is fit on the complete-case sample of those
    columns (used to put the IC complexity family on a single common IC sample,
    matching the manuscript's IC-vs-Conway-vs-AI-side comparison).
    """
    dv = "DV_BeliefChange_Specific"
    needcov = ["Pre_Belief_Specific"] + (["OpenendedResponseWordCount"] if covset == "B" else [])
    drop = [dv, col, "Pre_Belief_Specific"] + (common_cols or [])
    sub = df.dropna(subset=drop).copy()
    if covset == "B":
        sub = sub.dropna(subset=["OpenendedResponseWordCount"])
    if canon_only:
        sub = sub[sub["_canon"]]
    if len(sub) < 30:
        return None
    y = sub[dv].values
    mz = zs(sub[col].values)
    cov = [zs(sub[c].values) for c in needcov]
    # collinearity diagnostic: R^2(moderator ~ covariates)
    collin = sm.OLS(mz, sm.add_constant(np.column_stack(cov))).fit().rsquared
    Xlin = sm.add_constant(np.column_stack([mz] + cov))
    mlin = sm.OLS(y, Xlin).fit()
    if binary or sub[col].nunique() <= 2:
        return (len(sub), mlin.params[1], mlin.pvalues[1], np.nan, np.nan, np.nan, collin)
    mz2 = mz ** 2  # z(M)^2 headline scale
    Xq = sm.add_constant(np.column_stack([mz, mz2] + cov))
    mq = sm.OLS(y, Xq).fit()
    dr2 = mq.rsquared - mlin.rsquared
    return (len(sub), mlin.params[1], mlin.pvalues[1], mq.params[2], mq.pvalues[2], dr2, collin)


def run():
    df = load()
    n_age = "age" in df.columns and df["age"].notna().any()
    print(f"Loaded treatment N = {len(df)} (age column: {'present' if n_age else 'MISSING'})\n")

    for covset, desc in [("A", "SPEC A: covs = pre-belief only (14's minimal spec)"),
                         ("B", "SPEC B: covs = pre-belief + word count (headline spec)")]:
        print("=" * 78)
        print(desc)
        print("=" * 78)
        print(f"  {'Moderator':<32s} {'n':>5s} {'b_lin':>9s} {'p':>6s} "
              f"{'b_quad':>8s} {'p':>9s} {'dR2':>7s} {'collin':>7s}")
        # Complexity-family rows share a single common IC sample (so IC, Conway,
        # gpt-baseline and AI-side IC are compared on the same participants).
        IC_COMMON = ["IC_q400", "IC_conway"]
        rows = []
        flagged = []
        for col, label, cat, binary in MODS:
            if col not in df.columns:
                print(f"  {label:<32s}  -- column absent --")
                continue
            is_complex = cat.startswith("Complexity")
            common = IC_COMMON if is_complex else None
            # anchor EVERY row to the canonical Costello analysis sample so the
            # whole table is on one reproducible participant frame (IC = -1.99,
            # IH/AOT = the cited Study-1 N), differing only by item missingness.
            r = fit_mod(df, col, binary, covset, common_cols=common, canon_only=True)
            if r is None:
                continue
            n, bl, pl, bq, pq, dr2, collin = r
            rows.append((label, cat, dr2, bq, pq))
            degen = collin > 0.98  # near-perfect collinearity -> b_lin not identified
            if degen:
                flagged.append(label)
            bls = "DEGEN" if degen else f"{bl:+.2f}"
            bqs = f"{bq:+.2f}" if not np.isnan(bq) else "  --  "
            pqs = f"{pq:.3g}" if not np.isnan(pq) else "  -- "
            dr2s = f"{dr2:.4f}" if not np.isnan(dr2) else "  --  "
            print(f"  {label:<32s} {n:>5d} {bls:>9s} {pl:>6.2f} "
                  f"{bqs:>8s} {pqs:>9s} {dr2s:>7s} {collin:>7.3f}")
        if flagged:
            print(f"\n  [collinear b_lin (R^2(mod~cov) > 0.98), linear term NOT identified]: "
                  f"{', '.join(flagged)}")
        # IC vs best non-IC quadratic moderator
        ic = next((dr2 for lbl, cat, dr2, bq, pq in rows if lbl == "IC"), np.nan)
        noic = [(lbl, dr2) for lbl, cat, dr2, bq, pq in rows
                if not cat.startswith("Complexity") and cat != "Belief" and not np.isnan(dr2)]
        noic.sort(key=lambda t: -t[1])
        print(f"\n  q400 IC dR2 = {ic:.4f}")
        if noic:
            top_lbl, top = noic[0]
            print(f"  largest non-IC quadratic moderator: {top_lbl} dR2 = {top:.4f}")
            print(f"  ratio IC / {top_lbl} = {ic/top:.2f}x")
        print()


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        run()
    text = buf.getvalue()
    sys.stdout.write(text)
    (HERE / "note23b_competing_moderators_output.txt").write_text(text)
    sys.stdout.write(f"\nWrote {HERE / 'note23b_competing_moderators_output.txt'}\n")
