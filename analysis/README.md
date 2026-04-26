# Analysis pipeline

Each numbered script reproduces a specific section of the paper. All scripts read from `../data/` and print results to stdout (with optional `--output` flags for some); none require API keys or GPU access.

## Quick start

```bash
# Reproduce the headline (Costello inverted-U)
python 01_costello_analysis.py
```

## Script-to-section mapping

All analysis scripts run in seconds (under 1 minute even on a modest laptop). The only exception is `01_costello_analysis.py`, which includes bootstrap CIs and takes ~15 seconds.

| Script | Reproduces |
|---|---|
| `01_costello_analysis.py` | Costello primary fit, within-study replication, quintile analysis, discriminant validity, competing moderators (24-variable), IC × argument quality |
| `02_cheng_analysis.py` | Cheng boundary condition (SI Note 5a) |
| `03_salvi_analysis.py` | Salvi adversarial-debate boundary (SI Note 5b) |
| `06_absolute_change_engagement.py` | \|Δ\| persuasibility reanalysis (SI Note 9) |
| `07_sentence_completion_stability.py` | Cross-content stability (sentence-completion, *N* = 887, SI Note 4) |
| `08_scorer_validation.py` | Held-out validation on Suedfeld and Jakob (SI Note 2) |
| `note22_loso_enrichment.py` | Out-of-sample enrichment via leave-one-study-out (SI Note 22) — needs raw Costello for `StudyNumber` |
| `10_topic_fixed_effects.py` | Within-topic fixed-effects refit (SI Note 19) |
| `11_baseline_anchor.py` | Control-arm placebo anchoring (SI Note 9) |
| `12_quintile_demographics.py` | IH-quintile demographic composition (SI Note 23) |
| `13_text_model_shape_test.py` | Generic text-model shape test (SI Note 26) — **optional**, requires `OPENAI_API_KEY` if cache missing |
| `note03_text_vs_questionnaire.py` | Text-based vs questionnaire-based classifier on evaluative-vs-compliant belief change (SI Note 3) — needs raw Costello |
| `note06_engagement_independence.py` | Costello engagement-independence (SI Note 6) |
| `note07_boissin_engagement.py` | Within-Boissin engagement-independence (SI Note 7) |
| `note10_scorer_artifact.py` | Scorer-artifact / Suedfeld inverse-calibration test (SI Note 10) |
| `note11_zero_mass_sensitivity.py` | DV zero-mass sensitivity (SI Note 11) |
| `note12_pre_only_replication.py` | Pre-treatment-only IC replication (SI Note 12) |
| `note13_boissin_spec_curve.py` | Boissin specification-curve analysis (SI Note 13) |
| `note14_apex_shift_test.py` | Within-Costello apex-shift test (SI Note 14) |
| `note15_threshold_sensitivity.py` | Adverse-movement threshold sensitivity (SI Note 15) |
| `note16_spline_shape_test.py` | Spline vs quadratic shape test (SI Note 16) — GAM optional, falls back gracefully |
| `note17_ic_distribution.py` | Costello IC distribution vs Jakob/Suedfeld baselines (SI Note 17) |
| `note18_turnwise_ic_stability.py` | Turn-by-turn IC stability + cumulative-window refits (SI Note 18) |
| `note19_ai_side_ic.py` | AI-side argument complexity — Tables aiic_survival, complexity_vs_gap, topic_fe (SI Note 19) — needs raw Costello for the topic-FE block |
| `note20_surface_features.py` | Surface-feature ablation (SI Note 20) — uses bundled `costello_surface_features.csv` |
| `note21_reception_demand.py` | Reception-demand modulation (SI Note 21) — uses bundled `costello_demand_composite.csv` |
| `note23_incremental_validity.py` | Incremental validity Table tab:incremental_validity (SI Note 23) — needs raw Costello for full demographics |
| `note25_alt_constructs_discriminant.py` | Alternative-constructs discriminant validity, including validated text-IH (SI Note 25) |
| `note05c_tessler.py` | Tessler/Habermas null-IC moderation (SI Note 5c) |

## Data dependencies

Scripts split into two tiers based on what data they need:

**Tier 1 — runs out-of-the-box from bundled scored data** (no extra downloads):
- `01_costello_analysis.py` (headline + 24-moderator + within-study + quintile)
- `02_cheng_analysis.py`
- `03_salvi_analysis.py`
- `08_scorer_validation.py`

**Tier 2 — requires the raw Costello publication CSV** (`AllDataForPublication.PPI.8.28.24.csv`, ~4 MB, downloadable from [OSF gdkb7](https://osf.io/gdkb7/), placed at `../data/costello2024/Data 8.28.24/`):
- `06_absolute_change_engagement.py` (placebo arm fits)
- `10_topic_fixed_effects.py` (conspiracy-topic clustering needs raw text)
- `11_baseline_anchor.py` (control-arm scoring)
- `12_quintile_demographics.py` (full demographic columns)
- `13_text_model_shape_test.py` (also needs OpenAI embeddings cache; see below)
- `note03_text_vs_questionnaire.py` (needs `GPT_CoT_PlausibilityRating` and `AIProvidedInaccurateSummary`)
- `note19_ai_side_ic.py` (topic-FE block needs `conspiracyTheory` text; AI-side IC scores are bundled)
- `note22_loso_enrichment.py` (needs `StudyNumber`)
- `note23_incremental_validity.py` (needs `AgeYears`, `religion`, `genai_trust`, `Extremism`)

Tier-2 scripts will exit with a clear "ERROR: raw Costello publication data missing at … download from https://osf.io/gdkb7/" message if the file is absent.

**Tier 3 — requires additional external data**:
- `07_sentence_completion_stability.py` requires the cross-content stability dataset (*N* = 887, sentence-completion stems) cited in SI Note 4. Not redistributed; see paper SI Note 4 for source.
- `13_text_model_shape_test.py` additionally requires either the OpenAI embeddings cache (not bundled) or `OPENAI_API_KEY` set to re-fetch.

To re-score raw participant text from any source dataset (rather than use the bundled scored CSVs), see `../scoring/` and `../docs/SCORER_USAGE.md`.

## Expected headline output

After `python 01_costello_analysis.py`:

```
QUADRATIC MODERATION — Primary (Q400 logit-EV)
  n = 1782
  β_IC²            = -15.17, p < .001
  BF₁₀(quad/lin)   = 1086.4
  Bootstrap apex   = 2.76 IC, 95% CI [2.50, 3.02]
```

(Bayes factors are computed via BIC approximation; small numerical differences across BLAS / Python versions are expected.)
