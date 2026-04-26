"""Self-contained inference example for the IH-Scorer LoRA adapter.

Implements the **logit-EV decoding** used in the paper: a single forward pass
extracts the logits at the last position over the five letter tokens "A".."E",
applies softmax, and returns the expected value, affine-mapped to a continuous
1..6 scale for downstream regression.

This script uses **unsloth's FastModel** for loading, which is what the
training run used. Loading the LoRA via the same code path that produced the
training-time outputs avoids quantization-kernel drift between training and
inference.

Usage
-----
    pip install -U unsloth bitsandbytes accelerate
    # (PyTorch + CUDA 12.x must already be installed and matched to your GPU)

    from inference_example import score_texts
    scores = score_texts(["Some passage to score.", "Another text."])
    # → [3.7, 2.1]   floats in [1, 6]

The first call downloads the base model (`unsloth/Qwen3.5-27B`) from the Hub
and applies this LoRA. Base ≈ 17 GB on disk after 4-bit; first download takes
some minutes. Inference at 4-bit needs ≥24 GB VRAM at batch=4, seq_len=1200.

Hardware
--------
- 4-bit (default): single GPU with ≥24 GB VRAM.
- bf16 (load_in_4bit=False): single GPU with ≥56 GB VRAM, or multi-GPU.

Notes
-----
- The scoring scale is A (intellectually arrogant) through E (deeply humble),
  affine-mapped to a continuous [1, 6] scale for downstream regression:
  `score = 1 + (ev - 1) * 5 / 4`, where `ev` is the expected letter index
  in [1, 5].
- The adapter was trained with the `decomp` system prompt (3-sub-question
  decomposition); that exact prompt is hard-coded below. Do not change it
  without retraining.
- Texts longer than `max_seq_len` tokens are truncated from the *right*. For
  very long passages, score the first ~800 words rather than truncating
  arbitrarily.
- Padding is left-side (Qwen default). The last-position formula is
  padding-side-agnostic — do not "fix" it.

Vanilla-transformers fallback (untested)
----------------------------------------
If you cannot install unsloth, the LoRA can in principle be loaded with
`AutoModelForCausalLM` + `peft.PeftModel.from_pretrained` + `bitsandbytes`
quantisation. Use NF4 with bf16 compute and double-quant to match training.
This path was *not* validated; scores may differ from the reported results.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Defaults — override as kwargs if needed.
# ---------------------------------------------------------------------------
ADAPTER_DIR = str(Path(__file__).resolve().parent)  # this repo
MAX_SEQ_LEN = 1200
BATCH_SIZE = 4
LETTERS = list("ABCDE")

SYS_PROMPT = (
    "Rate the Intellectual Humility of the text. Consider in order: "
    "(1) Does the writer acknowledge specific limits of their own knowledge? "
    "(2) Are they open to revising views on evidence, not just in principle? "
    "(3) Do they engage opposing views fairly rather than rhetorically? "
    "Then return a single letter and nothing else: A (intellectually arrogant, "
    "none of the above) through E (deeply humble, all three)."
)


# ---------------------------------------------------------------------------
# Model loading (unsloth — matches training pipeline)
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
    LoRA in one shot.
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


def _letter_token_ids(tokenizer) -> List[int]:
    """Map letters "A".."E" to their last subword token id."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return [tok.encode(L, add_special_tokens=False)[-1] for L in LETTERS]


def _ev_to_score(ev: float) -> float:
    """Affine map A..E expected value (range [1, 5]) to a continuous [1, 6] scale."""
    return 1.0 + (ev - 1.0) * 5.0 / 4.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score_batch(
    model,
    tokenizer,
    texts: Sequence[str],
    letter_ids: List[int],
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[List[str], List[float], List[float]]:
    """One forward pass; returns (letter, ev_1to5, score_1to6) per text."""
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

    ids = torch.tensor(letter_ids, device=out.logits.device)

    letters, evs, scores = [], [], []
    for b in range(len(texts)):
        pos = last_pos[b].item()
        logits_5 = out.logits[b, pos, ids]              # shape (5,)
        probs = torch.softmax(logits_5, dim=0)
        ev = float(sum((i + 1) * probs[i].item() for i in range(5)))
        letters.append(LETTERS[int(torch.argmax(probs).item())])
        evs.append(ev)
        scores.append(_ev_to_score(ev))
    return letters, evs, scores


def score_texts(
    texts: Sequence[str],
    *,
    return_all: bool = False,
    adapter_dir: str = ADAPTER_DIR,
    load_in_4bit: bool = True,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Score a list of English texts on Intellectual Humility.

    Returns
    -------
    list[float]
        Continuous [1, 6] scores. Recommended for downstream regression /
        correlation work.
    tuple of (list[str], list[float], list[float])
        If return_all=True: (letters, ev_1to5, score_1to6). Letters in
        {A..E}, ev in [1, 5] continuous, score in [1, 6] continuous.

    Examples
    --------
    >>> scores = score_texts([
    ...     "I know what I'm talking about, unlike most people. Anyone who "
    ...     "disagrees hasn't done their research.",
    ...     "I've held this view for a while but I recognize there's a lot I "
    ...     "don't know. The strongest argument against it is real and I "
    ...     "can't fully rebut it.",
    ... ])
    >>> scores   # ~ [1.4, 5.2]
    """
    model, tokenizer = load_model(
        adapter_dir=adapter_dir,
        load_in_4bit=load_in_4bit,
        max_seq_len=max_seq_len,
    )
    letter_ids = _letter_token_ids(tokenizer)

    all_letters: List[str] = []
    all_evs: List[float] = []
    all_scores: List[float] = []
    for i in range(0, len(texts), batch_size):
        L, e, s = _score_batch(
            model, tokenizer, texts[i : i + batch_size],
            letter_ids, max_seq_len=max_seq_len,
        )
        all_letters.extend(L)
        all_evs.extend(e)
        all_scores.extend(s)

    if return_all:
        return all_letters, all_evs, all_scores
    return all_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    import argparse, json, sys

    p = argparse.ArgumentParser(
        description="Score texts on Intellectual Humility. "
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
    args = p.parse_args()

    if args.input:
        with open(args.input) as f:
            texts = [ln.rstrip("\n") for ln in f if ln.strip()]
    else:
        texts = [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    if not texts:
        print("No input texts.", file=sys.stderr); sys.exit(1)

    letters, evs, scores = score_texts(
        texts, return_all=True,
        load_in_4bit=not args.no_4bit, batch_size=args.batch,
    )

    out = sys.stdout if args.output == "-" else open(args.output, "w")
    for t, L, e, s in zip(texts, letters, evs, scores):
        rec = {
            "text": t[:200],
            "letter": L,
            "ev": round(e, 3),
            "score": round(s, 3),
        }
        print(json.dumps(rec, ensure_ascii=False), file=out)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    _cli()
