"""Self-contained inference example for the IC-Scorer Q400 LoRA adapter.

Implements the **logit-EV decoding** used in the paper: a single forward pass
extracts the logits at the last position over the seven score tokens "1".."7",
applies softmax, and returns the expected value E[IC] = Σ i · p_i.

This script uses **unsloth's FastModel** for loading, which is what the
training and validation runs used. Loading the LoRA via the same code path
that produced the validation numbers avoids quantization-kernel drift between
training and inference.

Usage
-----
    pip install -U unsloth bitsandbytes accelerate
    # (PyTorch + CUDA 12.x must already be installed and matched to your GPU)

    from inference_example import score_texts
    ev = score_texts(["Some passage to score.", "Another text."])
    # → [4.12, 2.07]   floats in [1, 7]

The first call downloads the base model (`unsloth/Qwen3.5-27B`) from the Hub
and applies this LoRA. Base ≈ 17 GB on disk after 4-bit; first download takes
some minutes. Inference at 4-bit needs ≥24 GB VRAM at batch=8, seq_len=1024.

Hardware
--------
- 4-bit (default): single GPU with ≥24 GB VRAM.
- bf16 (load_in_4bit=False): single GPU with ≥56 GB VRAM, or multi-GPU.

Notes
-----
- The scoring scale is anchored to the IC 1–7 system (Suedfeld scoring
  tradition).
- Texts longer than `max_seq_len` tokens are truncated from the *right*. For
  very long passages, score the abstract / first ~800 words rather than
  truncating arbitrarily.
- Padding is left-side (Qwen default). The last-position formula is
  padding-side-agnostic — do not "fix" it.

Vanilla-transformers fallback (untested)
----------------------------------------
If you cannot install unsloth, the LoRA can in principle be loaded with
`AutoModelForCausalLM` + `peft.PeftModel.from_pretrained` + `bitsandbytes`
quantisation. Use NF4 with bf16 compute and double-quant to match training.
This path was *not* validated; expect ICC drift.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Defaults — override as kwargs if needed.
# ---------------------------------------------------------------------------
ADAPTER_DIR = str(Path(__file__).resolve().parent)  # this repo
MAX_SEQ_LEN = 1024
BATCH_SIZE = 8

SYS_PROMPT = (
    "Return the Integrative Complexity of the input text and nothing else. "
    "Score the passage from 1 (single absolute view) to 7 (holistic synthesis) "
    "solely by judging how many distinct perspectives it differentiates and how "
    "explicitly it weaves those perspectives into an integrated whole."
)


# ---------------------------------------------------------------------------
# Model loading (unsloth — matches training/eval pipeline)
# ---------------------------------------------------------------------------
_CACHED: dict = {}


def load_model(
    adapter_dir: str = ADAPTER_DIR,
    load_in_4bit: bool = True,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Load (and cache) the base model + LoRA via unsloth's FastModel.

    Returns (model, tokenizer). Subsequent calls with the same args reuse
    the cached instance.

    `adapter_dir` is passed directly to FastModel.from_pretrained, which
    reads `adapter_config.json` to find the base model and applies the
    LoRA in one shot. This is the same code path that produced the
    validation ICC values.
    """
    key = (adapter_dir, load_in_4bit, max_seq_len)
    if key in _CACHED:
        return _CACHED[key]

    from unsloth import FastModel  # imported lazily so the module is usable
                                   # even when unsloth isn't installed yet

    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=max_seq_len,
        load_in_4bit=load_in_4bit,
    )
    FastModel.for_inference(model)

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    _CACHED[key] = (model, tokenizer)
    return model, tokenizer


def _score_token_ids(tokenizer) -> dict:
    """Map score "1".."7" → its last subword token id."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return {i: tok.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 8)}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score_batch(
    model,
    tokenizer,
    texts: Sequence[str],
    id_per_score: dict,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[List[int], List[float]]:
    """One forward pass; returns (greedy_argmax, logit_EV) for each text."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)

    prompts = []
    for txt in texts:
        msgs = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": txt},
        ]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

    enc = tok(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_seq_len,
    ).to(model.device)

    with torch.no_grad():
        out = model(**enc)

    # Padding-side-agnostic: index of last non-pad token per row.
    attn = enc["attention_mask"]
    seq_len = attn.shape[1]
    last_pos = seq_len - 1 - attn.flip(dims=[1]).argmax(dim=1)

    score_ids = torch.tensor(
        [id_per_score[i] for i in range(1, 8)], device=out.logits.device,
    )

    greedy, ev = [], []
    for b in range(len(texts)):
        pos = last_pos[b].item()
        logits_7 = out.logits[b, pos, score_ids]      # shape (7,)
        probs = torch.softmax(logits_7, dim=0)
        ev.append(float(sum((i + 1) * probs[i].item() for i in range(7))))
        greedy.append(int(torch.argmax(probs).item()) + 1)
    return greedy, ev


def score_texts(
    texts: Sequence[str],
    *,
    return_greedy: bool = False,
    adapter_dir: str = ADAPTER_DIR,
    load_in_4bit: bool = True,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Score a list of English texts on Integrative Complexity (1–7).

    Returns
    -------
    list[float]
        Logit-EV scores in [1, 7]. The continuous channel — recommended.
    list[int] (only if return_greedy=True)
        Discrete argmax in {1..7}. About 0.02 ICC weaker on validation; use
        only when an integer scale is required.

    Examples
    --------
    >>> ev = score_texts([
    ...     "All immigrants should be deported.",
    ...     "While there are real concerns about immigration, "
    ...     "the economy depends on labour mobility, and a humane policy "
    ...     "must balance border security with the rights of those fleeing "
    ...     "persecution.",
    ... ])
    >>> ev   # ~ [1.4, 5.1]
    """
    model, tokenizer = load_model(
        adapter_dir=adapter_dir,
        load_in_4bit=load_in_4bit,
        max_seq_len=max_seq_len,
    )
    id_per_score = _score_token_ids(tokenizer)

    all_greedy: List[int] = []
    all_ev: List[float] = []
    for i in range(0, len(texts), batch_size):
        g, e = _score_batch(
            model, tokenizer, texts[i : i + batch_size],
            id_per_score, max_seq_len=max_seq_len,
        )
        all_greedy.extend(g)
        all_ev.extend(e)

    if return_greedy:
        return all_ev, all_greedy
    return all_ev


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    import argparse, json, sys

    p = argparse.ArgumentParser(
        description="Score texts on Integrative Complexity. "
                    "NOTE: each line in --input/stdin is treated as one text. "
                    "Multi-line passages should be passed via the Python API "
                    "(`score_texts(...)`), not the CLI.",
    )
    p.add_argument("--input", "-i", help="Path to a file with one text per line. "
                                         "If omitted, reads stdin.")
    p.add_argument("--output", "-o", default="-", help="Output path; '-' = stdout.")
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit loading (uses bf16; needs ~56 GB VRAM).")
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--greedy", action="store_true", help="Also return integer argmax.")
    args = p.parse_args()

    if args.input:
        with open(args.input) as f:
            texts = [ln.rstrip("\n") for ln in f if ln.strip()]
    else:
        texts = [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    if not texts:
        print("No input texts.", file=sys.stderr); sys.exit(1)

    result = score_texts(
        texts,
        return_greedy=args.greedy,
        load_in_4bit=not args.no_4bit,
        batch_size=args.batch,
    )
    ev = result[0] if args.greedy else result
    greedy = result[1] if args.greedy else None

    out = sys.stdout if args.output == "-" else open(args.output, "w")
    for i, (t, e) in enumerate(zip(texts, ev)):
        rec = {"text": t[:200], "logit_score": round(e, 3)}
        if greedy is not None:
            rec["greedy"] = greedy[i]
        print(json.dumps(rec, ensure_ascii=False), file=out)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    _cli()
