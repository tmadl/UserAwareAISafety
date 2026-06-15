#!/usr/bin/env python3
"""verify_lengthfix_numbers.py — recompute every manuscript q400 number on the
length-bug-fixed data and emit an authoritative ledger. Runs the remaining
captures not yet logged by hand. Writes verify_lengthfix_output.txt.
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
from scipy import stats as sp

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REL = HERE.parent


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / np.nanstd(v, ddof=1)


def load(mod, fn):
    spec = importlib.util.spec_from_file_location(mod, HERE / fn)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run():
    n06 = load("n06", "06_absolute_change_engagement.py")
    df = n06.load_costello()

    # ---- |Delta| apex (treatment, adjusted, raw IC bootstrap)
    t = df[df.is_treatment == 1].dropna(
        subset=["ic_q400", "abs_change", "Pre_Belief_Specific",
                "OpenendedResponseWordCount"]).copy()
    ic = t.ic_q400.values
    yabs = t.abs_change.values
    pre = zs(t.Pre_Belief_Specific.values)
    wc = zs(t.OpenendedResponseWordCount.values)
    X = np.column_stack([ic, ic ** 2, pre, wc])
    rng = np.random.RandomState(1)
    ap = []
    for _ in range(2000):
        idx = rng.randint(0, len(yabs), len(yabs))
        m = sm.OLS(yabs[idx], sm.add_constant(X[idx])).fit()
        if m.params[2] < 0:
            ap.append(-m.params[1] / (2 * m.params[2]))
    ap = np.array(ap)
    m0 = sm.OLS(yabs, sm.add_constant(X)).fit()
    print(f"|Delta| apex = {-m0.params[1]/(2*m0.params[2]):.2f} "
          f"CI[{np.percentile(ap,2.5):.2f},{np.percentile(ap,97.5):.2f}]  (was 2.47 [2.17,2.68])")

    # ---- signed control + treatment raw IC^2 quad
    for arm, lab, old in [(1, "treat", "-4.05"), (0, "control", "-0.96 (p=.082)")]:
        s = df[df.is_treatment == arm].dropna(
            subset=["ic_q400", "DV_BeliefChange_Specific", "Pre_Belief_Specific",
                    "OpenendedResponseWordCount"])
        icr = s.ic_q400.values
        X2 = sm.add_constant(np.column_stack(
            [icr, icr ** 2, zs(s.Pre_Belief_Specific), zs(s.OpenendedResponseWordCount)]))
        m = sm.OLS(s.DV_BeliefChange_Specific.values, X2).fit()
        print(f"signed {lab} raw IC^2 b = {m.params[2]:+.2f}, p = {m.pvalues[2]:.3f}  (was {old})")

    # ---- adverse rates
    advt = (df[df.is_treatment == 1].DV_BeliefChange_Specific <= -5).mean()
    advc = (df[df.is_treatment == 0].DV_BeliefChange_Specific <= -5).mean()
    print(f"adverse rate: treatment {100*advt:.1f}% / control {100*advc:.1f}%  (was 12.0/12.2)")

    # ---- AI-side IC (note19) and reception (note21): just run and tee tails
    print("\n--- note19 AI-side (key lines) ---")
    try:
        n19 = load("n19", "note19_ai_side_ic.py")
    except SystemExit:
        pass
    print("\n--- note21 reception (key lines) ---")
    try:
        n21 = load("n21", "note21_reception_demand.py")
    except SystemExit:
        pass


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            run()
        except Exception as e:
            print("ERROR:", e)
    text = buf.getvalue()
    (HERE / "verify_lengthfix_output.txt").write_text(text)
    sys.stdout.write(text)
