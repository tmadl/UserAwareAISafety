# Using the IC and IH scorers on new text

Two LoRA adapters are released on HuggingFace under CC-BY-NC-4.0:

- **IC** (Integrative Complexity, primary scorer) — [`tmadl/IC-Qwen3.5-ORPO-400`](https://huggingface.co/tmadl/IC-Qwen3.5-ORPO-400)
- **IH** (Intellectual Humility, construct-validated) — [`tmadl/IH-Qwen3.5-ORPO-Guo`](https://huggingface.co/tmadl/IH-Qwen3.5-ORPO-Guo)

Both are LoRA adapters on `unsloth/Qwen3.5-27B` (Apache-2.0). The base model is fetched on first use; ≈17 GB on disk after 4-bit quantization.

## Hardware

- 4-bit (default): single GPU with ≥24 GB VRAM (A10, RTX 3090/4090, A100-40GB).
- bf16 (`load_in_4bit=False`): ≥56 GB VRAM, or multi-GPU.

## Input format

Both scoring scripts accept a JSONL file with one record per line:

```jsonl
{"participantId": "subject_001", "text": "I think the moon landing was real because..."}
{"participantId": "subject_002", "text": "Some people say X but actually..."}
```

The `text` field is the passage to score (max 1,200 tokens). Other fields are passed through to the output CSV.

## Output format

Both scripts produce a CSV with columns:

| Column | Meaning |
|---|---|
| `participantId` | Pass-through from input |
| `ic_qwenorpo400_logit` (IC) or `ih_guo_decomp_ckpt100_logit` (IH) | Continuous logit-EV score |
| `ic_qwenorpo400_argmax` (IC only) | Most-likely integer 1–7 |
| `pred_letter` (IH only) | Most-likely letter A–E |

## Scoring decoding

Both scorers use **logit-expected-value (logit-EV) decoding**. For IC, a single forward pass extracts logits at the last position over the seven score tokens "1"–"7", applies softmax, and computes E[score] = Σ k · p_k. For IH, the same procedure runs over A–E and is affine-mapped to a continuous 1–6 scale.

This avoids the discreteness artefacts of integer-only argmax decoding (see SI Appendix Note 2).

## Prompts

The exact prompts used for training and scoring are hard-coded in the inference scripts (`scoring/score_ic.py` and `scoring/score_ih.py`). **Do not change them without retraining** — prompt mismatch will degrade performance.

For reference, the IC scorer uses a Suedfeld–Jakob-style instruction asking the model to rate differentiation and integration on a 1–7 scale; the IH scorer uses a 3-sub-question decomposition ("decomp") prompt asking the model to assess epistemic openness, openness to revision, and fair engagement.

## Calibration notes

- **IC** is calibrated against the Suedfeld–Tetlock 1977 expert-coded corpus and the Jakob et al. 2022 naturalistic online-discourse corpus. Held-out validation: Suedfeld-155 ICC(3,1) = .704 (r = .757); Jakob-2275 ICC(3,1) = .797 (r = .802).
- **IH** is calibrated against the Guo 2024 EMNLP IH-marker Reddit corpus. Held-out Guo: r = .71.

Both scorers' absolute outputs should be interpreted relative to their training-distribution scale, not as universal trait units.

## Out of scope

The scorers are released for psychological / social-science research, persuasion / belief-change studies, computational text-analysis pipelines, and replication exercises. They are **not** intended for:

- Individual psychological profiling
- Targeted persuasion or manipulation
- Ranking people by intellectual character
- Surveillance, content moderation, or platform-governance decisions
- High-stakes evaluation of identified individuals (students, employees, applicants, defendants, patients)
- Clinical or forensic assessment
- Hiring or selection decisions
- Downstream commercial products (commercial use is excluded under CC-BY-NC-4.0)

See the model cards on HuggingFace for full intended-use, limitations, and dual-use statements.
