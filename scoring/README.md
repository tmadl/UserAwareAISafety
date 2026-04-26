# Scoring pipeline

These scripts run the published LoRA scorers from HuggingFace on new text. Both adapters are released under CC-BY-NC-4.0 and require a single GPU with ≥24 GB VRAM (4-bit) or ≥56 GB (bf16).

## Scripts

| Script | Adapter | Output |
|---|---|---|
| `score_ic.py` | [`tmadl/IC-Qwen3.5-ORPO-400`](https://huggingface.co/tmadl/IC-Qwen3.5-ORPO-400) | continuous IC score on [1, 7] |
| `score_ih.py` | [`tmadl/IH-Qwen3.5-ORPO-Guo`](https://huggingface.co/tmadl/IH-Qwen3.5-ORPO-Guo) | continuous IH score on [1, 6] (affine-mapped from logit-EV over A–E) |

Both scripts use the unsloth `FastModel` loader to match the training-time loading path; this avoids quantization-kernel drift between training and inference.

## Quick start

```bash
# Programmatic API
python -c "
from scoring.score_ic import score_texts
print(score_texts(['Some passage to score.', 'Another text.']))
# [4.12, 2.07]
"

# CLI
python scoring/score_ic.py --input my_texts.jsonl --output my_ic_scores.csv
python scoring/score_ih.py --input my_texts.jsonl --output my_ih_scores.csv
```

## Input / output

See `../docs/SCORER_USAGE.md` for the expected JSONL input format, output CSV columns, prompt details, calibration notes, and out-of-scope uses.

## Hardware requirements

- 4-bit (default): ≥24 GB VRAM (A10, RTX 3090/4090, A100-40GB).
- bf16: ≥56 GB VRAM, or multi-GPU.
- The base model `unsloth/Qwen3.5-27B` (≈17 GB on disk after 4-bit) is fetched on first use; first download takes several minutes.
- Scoring throughput depends on batch size and sequence length; see the HF model card at `tmadl/IC-Qwen3.5-ORPO-400` for the full hardware-requirements summary.

## Reproducing paper IC scores

The bundled data files under `../data/ic_qwen3orpo400/` are the IC scores the paper analyses use. To verify reproducibility:

```bash
# Re-score Costello pre-treatment text
python scoring/score_ic.py \
    --input ../data/costello2024/texts_for_scoring.jsonl \
    --output /tmp/costello_rescored.csv

# Compare with the bundled scores
python -c "
import pandas as pd
old = pd.read_csv('../data/ic_qwen3orpo400/costello_texts_for_scoring_initial_qwenorpo400.csv')
new = pd.read_csv('/tmp/costello_rescored.csv')
m = old.merge(new, on='participantId', suffixes=('_bundled', '_rescored'))
print((m['ic_qwenorpo400_logit_bundled'] - m['ic_qwenorpo400_logit_rescored']).abs().describe())
"
```

Results should match to within numerical precision (logit-EV scoring is deterministic at temperature 0).
