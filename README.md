# UserAwareAISafety

Reproducibility code, derived scored data, figure scripts, and scorer utilities for:

> Madl, T. (2026). *Text-measured cognitive complexity predicts belief revision in AI persuasion.* PsyArXiv preprint / manuscript under review. <https://osf.io/preprints/psyarxiv/mdxvs_v3>

**What this shows.** AI-persuasion effects, often reported as uniform across users, are not. This reanalysis finds that a cognitive style measurable from text people already write — *integrative complexity* — predicts who revises their beliefs after AI debunking (a theory-predicted inverted-U) and which users bear a disproportionate share of adverse, belief-strengthening movement. It illustrates how a **user-side** signal can complement output-level (toxicity, factuality, sycophancy) AI-safety evaluation.
This repository is primarily the reproducibility package for the Costello AI-persuasion reanalysis. It also serves as a public code and measurement-tooling entry point for the broader research programme **Human Cognitive Autonomy in AI Interaction**, which studies how AI dialogue affects users' reflective agency and reasoning processes.

The reproducibility package contains the analysis pipeline, derived score tables, figure scripts, and scoring scripts needed to reproduce the main-text empirical claims and SI Appendix robustness checks. A coverage matrix in `docs/REPRODUCIBILITY.md` lists each SI Note's status; the numerical Notes covered by the matrix reproduce from this repository, with some requiring the raw Costello publication CSV from Dryad for participant-level fields not redistributed here. Model weights for the released scorers are hosted separately on Hugging Face.

## Start here

| Goal | Where to go |
|---|---|
| Reproduce the Costello reanalysis / IC paper | Start with [Quick start](#quick-start), then [Reproducing each paper claim](#reproducing-each-paper-claim). |
| Check which scripts reproduce which claims | See `docs/REPRODUCIBILITY.md`. |
| Score new text with the IC scorer or reproduce the paper-era IH checks | See [Scoring new text](#scoring-new-text) and `docs/SCORER_USAGE.md`. For new IH analyses, prefer `IH-scorer-v2` on Hugging Face. |
| Find the released scorer weights / model cards | See [Released research scorers](#released-research-scorers-used-here). |
| Understand the broader research programme | See [Programme context](#programme-context). |
| Cite the paper, code, or models | See [Citation](#citation). |

## Current AI safety work

This repository is a reproducibility artifact for a user-aware AI persuasion study. For my current technical AI-safety tooling work, see:

* [`distill-lint`](https://github.com/tmadl/distill-lint): a carrier-local audit tool for vocabulary-channel subliminal leakage in distilled language models. Given a student and its exact base model, it flags anomalously elevated tokens, tests whether the elevation appears readout-removable, and can scrub a confirmed-unwanted token with a self-check and rollback.
* Paper: [*Channel Location Constrains the Auditability of Subliminal Learning*](https://arxiv.org/abs/2606.22019), preprint under review at TMLR.

## About the paper

This paper reanalyses Costello et al.'s 2024 conversational-AI conspiracy-debunking experiment, in which 1,782 participants discussed personally held conspiracy beliefs with GPT-4. It asks whether **integrative complexity** (IC)—a text-measured cognitive-style signal capturing how people differentiate and integrate competing perspectives—predicts who revises their beliefs after AI persuasion.

### Headline result

IC moderates belief change in an **inverted-U** pattern: belief revision is largest at mid-sample IC and declines at both low and high IC. Across the IC range, fitted revision spans **about five points peak-to-trough** on the 0–100 belief scale — roughly 43% of the sample's average 11.8-point debunking effect, marking a tail-concentrated moderator rather than a uniform shift.

Technical headline (primary Q400 logit-EV scorer, $N = 1{,}782$):

- $\beta_{\text{IC}^2} = -1.99$ on the headline $z(\text{IC})^2$ scale (equivalently $-15.28$ on $z(\text{IC}^2)$, the scale the reproduction scripts print, and $-4.08$ on raw $\text{IC}^2$ — same fit, $p$, and BF)
- $\text{BF}_{10} = 1{,}205$
- $p < .001$

This matches the inverted-U pattern predicted by McGuire's reception–yielding trade-off under reception-demanding, evidence-dense persuasion. Low-IC pre-treatment texts are associated with reduced revision under dense, multi-step evidence; high-IC pre-treatment texts are associated with attenuated revision, consistent with greater yielding resistance or integration of new evidence into an already structured belief system. Mid-IC contexts revise most.

The result is robust across IC-scoring variants, within-study refits, demographic adjustment, topic controls, and discriminant tests against nearby constructs, including intellectual humility, actively open-minded thinking, need for cognition, and open-mindedness.

### Practical significance

The effect is modest in aggregate variance explained, as expected for an individual-difference moderator whose relevance concentrates at the tails. That tail concentration is the practically meaningful part: the bottom IC quintile holds **24%** (23.9%) of post-conversation belief-strengthening (adverse-movement) cases but only **14%** of large belief revisions, so **86%** of large revisions fall outside it.

This is a *population-level* concentration, not an individual-risk model. Out-of-sample discrimination of adverse movement reaches **AUC = 0.69** for the full paper-specification predictors, but that is carried almost entirely by pre-treatment belief strength; IC-only models are near chance (**AUC ≈ 0.44–0.56**). The pattern is therefore descriptive — a candidate for population-level auditing and for studying support mechanisms that preserve reflection under specified task conditions, not for assigning users to exclusion, gating, profiling, or persistent labels.

### Scope and boundary conditions

Across four additional conversational-AI datasets, IC moderation appears bounded by task structure. The Costello inverted-U appears in an evidence-dense, personally held, directional debunking dialogue. In the shorter Boissin debunking paradigm, IC shows the predicted yielding-resistance pattern rather than the full inverted-U. In paradigms lacking either a directional persuasion target or a personally held belief, the predicted Costello-style pattern is not recovered.

## Programme context

This repository anchors one empirical component of a broader programme on
Human Cognitive Autonomy in AI Interaction: when AI dialogue supports users'
reflective agency, and when it instead narrows, substitutes for, or weakens
users' own reasoning processes.

The present repository implements the reproducibility package for the Costello
AI-persuasion reanalysis. It does not attempt to reproduce or document the broader
theoretical framework, which is developed in a separate manuscript currently under review.

Related scorer model cards are linked below for transparency and reuse. Only
the IC scorer is required for the main Costello reproduction; the IH scorers
are included because they are used in construct-neighbour and reproducibility
checks. Other programme-related instruments are documented separately and are
not required to reproduce this paper.

## What's in here

| Directory | Contents |
|---|---|
| `analysis/` | Numbered Python scripts (`01`–`13`) that reproduce each main-text section and SI Note from the bundled scored datasets. |
| `scoring/` | Inference scripts for the IC scorer and legacy IH reproducibility scorer; loads weights from Hugging Face, scores text, writes CSVs. |
| `data/` | Per-dataset scored CSVs and validation data. **Raw participant text from the source datasets is not redistributed**; pointers below. |
| `docs/` | Reproducibility walk-through, scorer-usage notes, data provenance. |
| `figures/` | Figure-generating scripts; figures regenerate from data. |

## Released research scorers used here

The scorer weights used by this repository are hosted separately on Hugging Face.

Exact reproduction of the Costello / IC paper uses the IC scorer and, for some
construct-neighbour checks, the legacy IH reproducibility scorer. The legacy IH
scorer is retained for reproducibility only. For new IH analyses outside this
paper, use `IH-scorer-v2`.

| Scorer | Role in this repo |
|---|---|
| [`IC-Qwen3.5-ORPO-400`](https://huggingface.co/tmadl/IC-Qwen3.5-ORPO-400) | Required for the Costello / IC paper. |
| [`IH-Qwen3.5-ORPO-Guo`](https://huggingface.co/tmadl/IH-Qwen3.5-ORPO-Guo) | Frozen legacy IH scorer used for exact reproduction of paper-era construct-neighbour checks. |
| [`IH-scorer-v2`](https://huggingface.co/tmadl/IH-scorer-v2) | Recommended current IH scorer for new analyses outside exact reproduction. |

Full validation coefficients, training details, intended use, and limitations
are documented in the linked Hugging Face model cards. This repository
reproduces only the Costello / IC analyses.

### Measurement-status note

These scorers estimate expressed properties of passages under specified scoring
conditions. They should not be used as stable labels for people. In particular,
they are not suitable for diagnosis, individual-level profiling, eligibility
decisions, psychological targeting, access restriction, third-party reporting,
or deployment as standalone safety filters.

## Quick start

```bash
git clone https://github.com/tmadl/UserAwareAISafety.git
cd UserAwareAISafety
pip install -r requirements.txt
python reproduce_headline.py
```

This 30-second sanity check prints the headline result:

```
Costello inverted-U headline (Q400 logit-EV, paper-spec model)
  n                    = 1,782
  beta_IC^2 (z(IC)^2)  = -1.99    (paper headline: -1.99)
  beta_IC^2 (z(IC^2))  = -15.28   (same fit, other scale)
  p_IC^2               = 3.35e-06
  BF10(quad)           = 1,205.4
  apex IC              = 2.76 [2.50, 3.03]
```

For the full primary analysis (within-study replication, 24-moderator comparison, quintiles), run `python analysis/01_costello_analysis.py`.

## Reproducing each paper claim

For a 30-second sanity check on the main-text headline:

```bash
python reproduce_headline.py
# Costello inverted-U headline (Q400 logit-EV, paper-spec model)
#   beta_IC^2 (z(IC)^2) = -1.99   (= -15.28 on z(IC^2))   p = 3.35e-6   BF10 = 1,205   apex IC = 2.76 [2.50, 3.03]
```

To regenerate the main-text figures from data:

```bash
python figures/fig1_headline.py             # Fig 1: inverted-U
python figures/fig2_costello_deepdive.py    # Fig 2: enrichment + leave-one-study-out (LOSO) operating point
```

Each numbered script in `analysis/` corresponds to a section of the paper. See `docs/REPRODUCIBILITY.md` for the full mapping (script → main-text section / SI Note).

| Script | Reproduces |
|---|---|
| `reproduce_headline.py` | Main-text headline (Costello inverted-U: β, p, BF, apex) |
| `01_costello_analysis.py` | Costello primary + within-study + quintile + 24-moderator |
| `02_cheng_analysis.py` | Cheng boundary condition (SI Note 5a) |
| `03_salvi_analysis.py` | Salvi boundary condition (SI Note 5b) |
| `06_absolute_change_engagement.py` | \|Δ\| persuasibility reanalysis (SI Note 9) |
| `07_sentence_completion_stability.py` | Cross-content stability, *N* = 887 (SI Note 4) |
| `08_scorer_validation.py` | Held-out validation on Suedfeld and Jakob (SI Note 2) |
| `note22_loso_enrichment.py` | Out-of-sample enrichment / LOSO (SI Note 22) |
| `10_topic_fixed_effects.py` | Within-topic fixed-effects refit (SI Note 19) |
| `11_baseline_anchor.py` | Control-arm placebo anchoring (SI Note 9) |
| `12_quintile_demographics.py` | IC-quintile demographic composition (SI Note 23) |
| `13_text_model_shape_test.py` | Generic text-model shape test (SI Note 26) — optional, requires `OPENAI_API_KEY` to refetch embeddings |
| `figures/fig1_headline.py` | Main-text Figure 1 (Costello inverted-U) |
| `figures/fig2_costello_deepdive.py` | Main-text Figure 2 (enrichment + LOSO) |

## Scoring new text

To score new text with the published IC or IH scorers:

```bash
# Requires GPU with ≥24 GB VRAM (4-bit) or ≥56 GB (bf16)
python scoring/score_ic.py --input my_texts.jsonl --output my_ic_scores.csv
python scoring/score_ih.py --input my_texts.jsonl --output my_ih_scores.csv
```

**Note**: The bundled scripts reproduce the paper-era IC and legacy IH scoring paths. For new IH analyses, use the [`IH-scorer-v2`](https://huggingface.co/tmadl/IH-scorer-v2) Hugging Face inference code rather than `scoring/score_ih.py`.

See `docs/SCORER_USAGE.md` for input format, hardware notes, and the exact prompts.

## Data provenance

The repository ships **derivative scored data only**. Raw participant text and original analysis files come from the source datasets; download from the original repositories and place in `data/<dataset>/raw/` if you want to re-score from scratch.

| Dataset | Source | Redistribution |
|---|---|---|
| Costello et al. 2024 | [DRYAD — DOI 10.5061/dryad.v6wwpzh4h](https://datadryad.org/dataset/doi:10.5061/dryad.v6wwpzh4h) | Bundled: scored derivatives + filtered analysis files. Not bundled: raw `AllDataForPublication.PPI.8.28.24.csv` from the Dryad archive. |
| Boissin et al. 2025 | [OSF — wyvxf](https://osf.io/wyvxf/) | Bundled: scored derivatives + analysis files. Not bundled: raw participant data. |
| Cheng et al. 2026 | Original publication | Bundled: scored derivatives + analysis files. |
| Salvi et al. 2025 | [`frasalvi/debategpt`](https://huggingface.co/datasets/frasalvi/debategpt) | Bundled: scored derivatives. |
| Tessler et al. 2024 | *Science* data appendix | Bundled: scored derivatives. |
| Suedfeld-155 | [Tetlock & Suedfeld corpus](https://github.com/.../suedfeld) | Bundled: held-out validation scores. |
| Jakob-2275 | [Jakob et al. 2022 corpus](https://github.com/.../jakob2022) | Bundled: cross-validation scores. |
| Guo-EMNLP IH labels | [Guo 2024](https://github.com/xiaobo-guo/The-Computational-Anatomy-of-Humility-Modeling-Intellectual-Humility-in-Online-Public-Discourse) | Not bundled (used only in scorer training; weights on HF). |

## Citation

If you use this code, models, or scored data, please cite both the paper and the scorer model cards:

```bibtex
@misc{madl2026cognitive,
  author       = {Madl, Tamas},
  title        = {Text-measured cognitive complexity predicts belief revision in AI persuasion},
  year         = {2026},
  howpublished = {PsyArXiv preprint},
  url          = {https://osf.io/preprints/psyarxiv/mdxvs_v3},
  note         = {Reproducibility: https://github.com/tmadl/UserAwareAISafety}
}
```

If you use a scorer, cite the corresponding Hugging Face model card and version in addition to the paper.

## Licensing

- **Code** (`analysis/`, `scoring/`, `figures/`, repo-level scripts): MIT — see [`LICENSE`](LICENSE).
- **Derivative scored data** (`data/`): CC-BY-4.0 — see [`LICENSE-DATA`](LICENSE-DATA). Original-dataset redistribution rights are governed by the source-dataset licences (see Data provenance above).
- **Model weights**: CC-BY-NC-4.0, on the linked HuggingFace pages.

## Contact

Tamas Madl — `tamas.madl@ofai.at`
Austrian Research Institute for Artificial Intelligence (OFAI)
