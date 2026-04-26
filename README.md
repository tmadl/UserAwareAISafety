# UserAwareAISafety

Reproducibility code and data for:

> Madl, T. (2026). *Text-measured cognitive complexity predicts belief revision in AI persuasion.* (Manuscript under review.)

This repository contains the analysis pipeline, derivative scored data, and scoring scripts needed to reproduce the main-text empirical claims and the SI Appendix robustness checks. A coverage matrix in `docs/REPRODUCIBILITY.md` lists each SI Note's status; all numerical Notes reproduce from this repository (some require downloading the raw Costello publication CSV from OSF for participant-level fields not redistributed here). Model weights for the two fine-tuned LoRA scorers are released separately on HuggingFace.

## About the paper

This paper reanalyses Costello et al.'s 2024 conversational-AI conspiracy-debunking experiment, in which 1,782 participants discussed personally held conspiracy beliefs with GPT-4. It asks whether **integrative complexity** (IC)—a text-measured cognitive-style signal capturing how people differentiate and integrate competing perspectives—predicts who revises their beliefs after AI persuasion.

### Headline result

IC moderates belief change in an **inverted-U** pattern. Belief revision is largest at mid-sample IC and declines at both low and high IC:

- $\beta_{\text{IC}^2} = -15.17$
- $\text{BF}_{10} = 1{,}086$
- $p < .001$

This matches McGuire's reception–yielding trade-off. Low-IC users appear less able to receive dense, multi-step evidence; high-IC users appear better able to integrate new evidence into an already structured belief system without yielding. Mid-IC users revise most.

The result is robust across scorer choice, within-study refits, demographic adjustment, topic controls, and discriminant tests against nearby constructs (intellectual humility scored both via rubric prompting and via a construct-validated Guo-EMNLP-anchored text scorer; actively open-minded thinking; need for cognition; open-mindedness).

### Practical significance

The effect is modest in aggregate variance explained, as expected for an individual-difference moderator concentrated at the tails. But the tail pattern is practically meaningful: flagging only the lowest-IC quintile for adaptive pacing and comprehension safeguards would preserve **86%** of substantial belief revisions while capturing **24%** of post-conversation belief-strengthening cases. Leave-one-study-out performance gives **AUC = 0.69**.

### Scope and boundary conditions

Across four additional conversational-AI datasets, the effect appears bounded: IC moderates AI-mediated belief change when both conditions hold:

1. the AI has a directional persuasion target, and
2. the belief is personally held.

That pattern appears in the Costello and Boissin debunking datasets, but not in adversarial debate on assigned positions, interpersonal-conflict validation, or consensus-seeking deliberation.

## What's in here

| Directory | Contents |
|---|---|
| `analysis/` | Numbered Python scripts (`01`–`13`) that reproduce each main-text section and SI Note from the bundled scored data. |
| `scoring/` | Inference scripts for the two fine-tuned scorers; loads weights from HuggingFace, scores text, writes CSVs. |
| `data/` | Per-dataset scored CSVs (derivative works) and validation data. **Raw participant text from the source datasets is not redistributed**; pointers below. |
| `docs/` | Reproducibility walk-through, scorer-usage notes, data provenance. |
| `figures/` | Figure-generating scripts (figures themselves regenerate from data). |

## Released model weights

| Scorer | HuggingFace | Held-out validation |
|---|---|---|
| Integrative Complexity (primary) | [`tmadl/IC-Qwen3.5-ORPO-400`](https://huggingface.co/tmadl/IC-Qwen3.5-ORPO-400) | Suedfeld-155: ICC(3,1) = .704, r = .757; Jakob-2275: ICC(3,1) = .797, r = .802 |
| Intellectual Humility (construct-validated) | [`tmadl/IH-Qwen3.5-ORPO-Guo`](https://huggingface.co/tmadl/IH-Qwen3.5-ORPO-Guo) | Guo 2024 EMNLP held-out: r = .71 |

Both adapters are CC-BY-NC-4.0; LoRA on `unsloth/Qwen3.5-27B` (Apache-2.0).

## Quick start

```bash
git clone https://github.com/tmadl/UserAwareAISafety.git
cd UserAwareAISafety
pip install -r requirements.txt
python analysis/01_costello_analysis.py
```

The Costello analysis script prints the headline result:

```
QUADRATIC MODERATION — Primary (Q400 logit-EV)
  n = 1782
  β_IC²            = -15.17, p < .001
  BF₁₀(quad/lin)   = 1086.4
  bootstrap apex   = 2.76 IC, 95% CI [2.50, 3.02]
```

## Reproducing each paper claim

Each numbered script in `analysis/` corresponds to a section of the paper. See `docs/REPRODUCIBILITY.md` for the full mapping (script → main-text section / SI Note).

| Script | Reproduces |
|---|---|
| `01_costello_analysis.py` | Costello primary + within-study + quintile + discriminant validity |
| `02_cheng_analysis.py` | Cheng boundary condition (SI Note 5a) |
| `03_salvi_analysis.py` | Salvi boundary condition (SI Note 5b) |
| `06_absolute_change_engagement.py` | \|Δ\| persuasibility reanalysis (SI Note 9) |
| `07_sentence_completion_stability.py` | Cross-content stability, *N* = 887 (SI Note 4) |
| `08_scorer_validation.py` | Held-out validation on Suedfeld and Jakob (SI Note 2) |
| `09_detection_proxy.py` | Out-of-sample enrichment / screening (SI Note 22) |
| `10_topic_fixed_effects.py` | Within-topic fixed-effects refit (SI Note 19) |
| `11_baseline_anchor.py` | Control-arm placebo anchoring (SI Note 9) |
| `12_quintile_demographics.py` | IH-quintile demographic composition (SI Note 23) |
| `13_text_model_shape_test.py` | Generic text-model shape test (SI Note 26) — optional, requires `OPENAI_API_KEY` to refetch embeddings |

## Scoring new text

To score new text with the published IC or IH scorers:

```bash
# Requires GPU with ≥24 GB VRAM (4-bit) or ≥56 GB (bf16)
python scoring/score_ic.py --input my_texts.jsonl --output my_ic_scores.csv
python scoring/score_ih.py --input my_texts.jsonl --output my_ih_scores.csv
```

See `docs/SCORER_USAGE.md` for input format, hardware notes, and the exact prompts.

## Data provenance

The repository ships **derivative scored data only**. Raw participant text and original analysis files come from the source datasets; download from the original repositories and place in `data/<dataset>/raw/` if you want to re-score from scratch.

| Dataset | Source | Redistribution |
|---|---|---|
| Costello et al. 2024 | [OSF — gdkb7](https://osf.io/gdkb7/) | Bundled: scored derivatives + filtered analysis files. Not bundled: raw `AllDataForPublication.PPI.8.28.24.csv`. |
| Boissin et al. 2025 | [OSF — wyvxf](https://osf.io/wyvxf/) | Bundled: scored derivatives + analysis files. Not bundled: raw participant data. |
| Cheng et al. 2026 | Original publication | Bundled: scored derivatives + analysis files. |
| Salvi et al. 2025 | [`frasalvi/debategpt`](https://huggingface.co/datasets/frasalvi/debategpt) | Bundled: scored derivatives. |
| Tessler et al. 2024 | *Science* data appendix | Bundled: scored derivatives. |
| Suedfeld-155 | [Tetlock & Suedfeld 1977 corpus](https://github.com/.../suedfeld) | Bundled: held-out validation scores. |
| Jakob-2275 | [Jakob et al. 2022 corpus](https://github.com/.../jakob2022) | Bundled: cross-validation scores. |
| Guo-EMNLP IH labels | [Guo 2024](https://github.com/xiaobo-guo/...) | Not bundled (used only in scorer training; weights on HF). |

## Citation

If you use this code, models, or scored data, please cite both the paper and the scorer model cards:

```bibtex
@unpublished{madl2026cognitive,
  author = {Madl, Tamas},
  title  = {Text-measured cognitive complexity predicts belief revision in AI persuasion},
  year   = {2026},
  note   = {Manuscript under review.
            Reproducibility: https://github.com/tmadl/UserAwareAISafety}
}
```

## Licensing

- **Code** (`analysis/`, `scoring/`, `figures/`, repo-level scripts): MIT — see [`LICENSE`](LICENSE).
- **Derivative scored data** (`data/`): CC-BY-4.0 — see [`LICENSE-DATA`](LICENSE-DATA). Original-dataset redistribution rights are governed by the source-dataset licences (see Data provenance above).
- **Model weights**: CC-BY-NC-4.0, on the linked HuggingFace pages.

## Contact

Tamas Madl — `tamas.madl@ofai.at`
Austrian Research Institute for Artificial Intelligence (OFAI)
