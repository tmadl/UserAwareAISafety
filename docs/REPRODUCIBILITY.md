# Reproducibility walk-through

Every numerical claim in the paper main text and SI Appendix is reproducible from this repository plus the two LoRA scorers on HuggingFace. This document maps each script to the section / Note it reproduces, lists the expected runtime, and notes any optional external dependencies.

## Setup

```bash
git clone https://github.com/tmadl/UserAwareAISafety.git
cd UserAwareAISafety
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. The analysis scripts run on CPU; the scoring scripts (`scoring/score_ic.py`, `scoring/score_ih.py`) require a GPU with ≥24 GB VRAM for 4-bit inference.

## Running the analysis pipeline

To reproduce a section of the paper, run the corresponding numbered script:

| Script | Reproduces (main text) | Reproduces (SI) |
|---|---|---|
| `01_costello_analysis.py` | Costello headline (β = −15.17, BF = 1086, apex = 2.76); within-study replication; quintile analysis; 24-moderator comparison | Notes 6, 9, 18, 23, 24, 25 (partial) |
| `02_cheng_analysis.py` | Cheng boundary condition | Note 5a |
| `03_salvi_analysis.py` | Salvi boundary condition | Note 5b |
| `06_absolute_change_engagement.py` | \|Δ\| persuasibility reanalysis | Note 9 |
| `07_sentence_completion_stability.py` | Cross-content stability *N* = 887 | Note 4 |
| `08_scorer_validation.py` | Held-out validation on Suedfeld + Jakob | Note 2 |
| `09_detection_proxy.py` | Out-of-sample enrichment / leave-one-study-out | Note 22 |
| `10_topic_fixed_effects.py` | Within-topic fixed-effects refit | Note 19 |
| `11_baseline_anchor.py` | Control-arm placebo anchoring | Note 9 |
| `12_quintile_demographics.py` | IH-quintile demographic composition | Note 23 |
| `13_text_model_shape_test.py` | Generic text-model shape test (optional) | Note 26 |
| `note06_engagement_independence.py` | Costello engagement-independence | Note 6 |
| `note10_scorer_artifact.py` | Suedfeld inverse-calibration scorer-artifact test | Note 10 |
| `note11_zero_mass_sensitivity.py` | DV zero-mass sensitivity | Note 11 |
| `note12_pre_only_replication.py` | Pre-treatment-only IC replication | Note 12 |
| `note15_threshold_sensitivity.py` | Adverse-movement threshold sensitivity | Note 15 |
| `note16_spline_shape_test.py` | Spline / cubic shape test | Note 16 |
| `note17_ic_distribution.py` | IC distribution vs Jakob / Suedfeld baselines | Note 17 |
| `note18_turnwise_ic_stability.py` | Turn-by-turn IC stability + cumulative-window refits | Note 18 |
| `note20_surface_features.py` | Surface-feature ablation (stub; needs raw Costello) | Note 20 |
| `note21_reception_demand.py` | Reception-demand modulation (stub; needs raw Costello) | Note 21 |
| `note25_alt_constructs_discriminant.py` | Alternative-constructs discriminant validity | Note 25 |

All analysis scripts run in seconds; `01_costello_analysis.py` and `note12_pre_only_replication.py` include bootstrap CIs and take ~15 seconds. No GPU required for analysis; scoring scripts in `scoring/` need a GPU.

## Optional external dependencies

`analysis/13_text_model_shape_test.py` (SI Note 26) compares the inverted-U against four text-encoder baselines (TF-IDF, MiniLM, OpenAI text-embedding-3-small, OpenAI text-embedding-3-large). The OpenAI embeddings are not bundled; to reproduce the OpenAI panels, set `OPENAI_API_KEY` in your environment and let the script re-fetch them (the embedding API call is the only network dependency; cost is small). The TF-IDF and MiniLM panels run without any external API. The result is a single robustness check; the headline IC moderation is independent of this script.

## SI Note coverage from this repository

The public scripts here reproduce the main-text headline and every numerical SI Appendix Note.

Legend: ✅ reproduces from bundled data · 📥 reproduces, requires extra data download · ⚠️ partial coverage · N/A narrative.

| SI Note | Topic | Public-repo coverage |
|---|---|---|
| 1 | Theoretical framework (reception–yielding) | N/A — narrative |
| 2 | IC Measurement / Q400 validation | ✅ `08_scorer_validation.py` |
| 3 | Text vs questionnaire classification | 📥 `note03_text_vs_questionnaire.py` (needs raw Costello CSV) |
| 4 | Cross-content stability (*N* = 887 sentence completion) | 📥 `07_sentence_completion_stability.py` (needs sentence-completion dataset; see SI Note 4 source) |
| 5 | Boundary-condition datasets | ✅ `02_cheng_analysis.py` (5a), `03_salvi_analysis.py` (5b), `note05c_tessler.py` (5c); Boissin handled in Notes 12/13 |
| 6 | Costello engagement-independence | ✅ `note06_engagement_independence.py` |
| 7 | Within-Boissin engagement-independence | ✅ `note07_boissin_engagement.py` |
| 8 | Option-δ rejection (cross-dataset rank-order) | N/A — narrative |
| 9 | Persuasibility \|Δ\| reanalysis | 📥 `06_absolute_change_engagement.py`, `11_baseline_anchor.py` (need raw Costello) |
| 10 | Scorer-artifact test | ✅ `note10_scorer_artifact.py` |
| 11 | DV zero-mass sensitivity | ✅ `note11_zero_mass_sensitivity.py` |
| 12 | Pre-treatment-only IC replication | ✅ `note12_pre_only_replication.py` |
| 13 | Boissin specification curve | ✅ `note13_boissin_spec_curve.py` (288 of the SI's 324 cells; small-cell drops below n=30 minimum) |
| 14 | Within-Costello apex-shift test | ✅ `note14_apex_shift_test.py` |
| 15 | Threshold sensitivity (enrichment) | ✅ `note15_threshold_sensitivity.py` |
| 16 | Spline vs quadratic shape | ✅ `note16_spline_shape_test.py` (cubic OLS exact; GAM asymmetry diagnostic depends on pygam version) |
| 17 | Costello IC distribution vs Jakob baseline | ✅ `note17_ic_distribution.py` |
| 18 | Within-dialogue IC stability | ✅ `note18_turnwise_ic_stability.py` |
| 19 | AI-side argument complexity | 📥 `note19_ai_side_ic.py` (Tables 1–3 reproduce; topic-FE block uses bundled AI-side IC scores + raw Costello for `conspiracyTheory` text) |
| 20 | Surface-feature ablation | ✅ `note20_surface_features.py` (uses bundled `costello_surface_features.csv`) |
| 21 | Reception-demand modulation | ✅ `note21_reception_demand.py` (uses bundled `costello_demand_composite.csv`) |
| 22 | Out-of-sample enrichment / LOSO | 📥 `note22_loso_enrichment.py` (needs raw Costello for `StudyNumber`) |
| 23 | Incremental validity | 📥 `12_quintile_demographics.py` (quintile composition) + `note23_incremental_validity.py` (incremental-validity table; needs raw Costello for full demographics) |
| 24 | Falsification exposure | N/A — narrative |
| 25 | Discriminant validity (alternative constructs) | ✅ `note25_alt_constructs_discriminant.py` |
| 26 | Generic text-model shape test | ✅ `13_text_model_shape_test.py` (TF-IDF + MiniLM panels reproduce; OpenAI-embedding panels need cache or `OPENAI_API_KEY`) |

📥 Notes require the raw Costello publication CSV (downloadable from [OSF — gdkb7](https://osf.io/gdkb7/)). Costello's raw data is governed by the source publication's licence; we redistribute only derivative scored fields here, and 📥 Notes use participant-level fields (e.g., `GPT_CoT_PlausibilityRating`, `StudyNumber`, demographics) that are not in our derivative bundle. The relevant scripts print clear download instructions on first run.

All SI Notes that make numerical claims have a public reproduction script.

All main-text empirical claims and all numerically-tabulated SI Notes reproduce from this repository.

## Verifying you reproduced the headline

After `python analysis/01_costello_analysis.py`, the printed output should contain (within rounding):

```
QUADRATIC MODERATION — Primary (Q400 logit-EV)
  n = 1782
  β_IC²            = -15.17, p < .001
  BF₁₀(quad/lin)   = 1086.4
  Bootstrap apex   = 2.76 IC
  95% CI           = [2.50, 3.02]
```

If any number differs by more than a small rounding tolerance, please open an issue with the discrepancy and your environment details (Python version, package versions, OS).

## Re-scoring from raw text (optional)

To re-score from raw participant text rather than use the bundled scored CSVs:

1. Download the raw data from the source dataset's repository (see `docs/DATA_PROVENANCE.md`).
2. Convert to JSONL with one record per line: `{"participantId": "...", "text": "..."}`.
3. Run the IC scorer:
   ```bash
   python scoring/score_ic.py --input raw_texts.jsonl --output ic_scores.csv
   ```
4. The output CSV will have columns `participantId` and `ic_qwenorpo400_logit`. Place in the appropriate `data/ic_qwen3orpo400/` location and re-run the analysis script.

The IC scorer requires a single GPU with ≥24 GB VRAM (4-bit inference). Throughput depends on batch size and sequence length; see `scoring/score_ic.py` for the inference loop and the HF model card at `tmadl/IC-Qwen3.5-ORPO-400` for the full hardware-requirements summary.
