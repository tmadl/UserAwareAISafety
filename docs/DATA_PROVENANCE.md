# Data provenance

This repository ships **derivative scored data only**: the source datasets'
participant-level analysis files joined with new IC and IH score columns
produced by the LoRA scorers. Raw participant text and the original full
publication CSVs are not redistributed; download from the cited sources if
you wish to re-score from scratch.

## What is bundled

```
data/
├── costello2024/
│   ├── analysis_data.csv                            ← Costello primary analysis subset (filtered to treatment with scorable text)
│   ├── all_complexity_scores.csv                    ← gpt-4.1-mini IC scores (initial + concatenated text)
│   ├── control_ic_scores.csv                        ← gpt-4.1-mini IC scored on control conRestatement
│   ├── costello_controls_qwenorpo400.csv            ← Q400 IC scored on control conRestatement
│   ├── texts_for_scoring.jsonl                      ← per-participant pre-treatment + concatenated text
│   ├── multimodel/
│   │   └── claude-haiku-4.5_mean.csv                ← Anthropic Haiku cross-vendor scores
│   └── ih_aot_prototypes/
│       └── preds_ih_guo_only_decomp_ckpt100.csv     ← validated IH (HF model) scores on Costello
├── boissin2025/         ← Boissin scored derivatives + analysis files
├── cheng2006/           ← Cheng scored derivatives + analysis files
├── salvi2005/           ← Salvi scored derivatives + analysis files
├── tessler2024/         ← Tessler scored derivatives + analysis files
├── ic_qwen3orpo400/     ← Q400 primary scorer outputs across all five datasets
├── ic_validation/       ← Suedfeld + Jakob validation set scores
└── jakob_texts_for_scoring.csv  ← Jakob source texts (for re-scoring)
```

Total bundled size: ≈48 MB.

## What is not bundled (and where to get it)

| Source | What's missing | Where to get it |
|---|---|---|
| Costello et al. 2024 | Raw `AllDataForPublication.PPI.8.28.24.csv` (4,015 participants × 184 columns); replication R scripts; raw Qualtrics export | [OSF — gdkb7](https://osf.io/gdkb7/) |
| Boissin et al. 2025 | Raw participant text; original analysis data (`data.csv`, `dataForPublication.csv`); replication R scripts | [OSF — wyvxf](https://osf.io/wyvxf/) |
| Cheng et al. 2026 | Raw participant transcripts; original analysis files | Original publication's data repository (see paper) |
| Salvi et al. 2025 | Raw debate transcripts; HuggingFace dataset version | [`frasalvi/debategpt`](https://huggingface.co/datasets/frasalvi/debategpt) |
| Tessler et al. 2024 | Raw deliberation corpus; preference-ranking data | *Science* data appendix (see paper) |
| Suedfeld-155 | Raw expert-exemplar text; integrative-complexity coding manual | Tetlock & Suedfeld archives; see paper SI Note 2 |
| Jakob-2275 | Raw naturalistic online-discourse text | Jakob et al. 2022 corpus — see paper SI Note 2 |
| Guo-EMNLP IH labels | IH-marker labelled Reddit corpus | [Guo 2024 GitHub](https://github.com/xiaobo-guo/...) |
| OpenAI text embeddings | Cache for SI Note 26 text-model shape test (small) | Re-fetch via OpenAI API (set `OPENAI_API_KEY`) |

## How to integrate a freshly downloaded raw dataset

If you want to re-score the raw text from scratch:

1. Download the raw dataset from the source link above.
2. Place it under `data/<dataset>/raw/`.
3. Run the appropriate text-extraction step. For Costello:
   ```bash
   # Costello: text_initial column → JSONL for scoring
   python -c "
   import pandas as pd, json
   d = pd.read_csv('data/costello2024/raw/AllDataForPublication.PPI.8.28.24.csv', low_memory=False)
   d = d[d['ExperimentalCondition'] == 'Treatment']
   for _, r in d.iterrows():
       print(json.dumps({'participantId': r['participantId'], 'text': r['userResponse_combined']}))
   " > data/costello2024/raw_texts.jsonl
   ```
4. Re-score:
   ```bash
   python scoring/score_ic.py --input data/costello2024/raw_texts.jsonl \
       --output data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv
   ```
5. Re-run the analysis:
   ```bash
   python analysis/01_costello_analysis.py
   ```

## Licensing of the bundled derivative data

The new score columns produced by the LoRA scorers are licensed under
**CC-BY-4.0** — see `LICENSE-DATA`. Cite the paper when redistributing.

The original participant-text and outcome columns inherited from the source
datasets remain governed by the original-dataset licences. If you redistribute
files containing those columns, you must comply with the source-dataset terms.
