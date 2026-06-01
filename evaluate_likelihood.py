"""
Conditional log-likelihood evaluation of a LLaDA-style SFT checkpoint.

Implements Algorithm 3 / Eq. (6) from the LLaDA paper:

    LL(r_0 | p_0) = E_{l, r_l} [ (L / l) * sum_i 1[r_l^i = M] log p_theta(r_0^i | p_0, r_l) ]

    where L is the response length, l ~ Uniform{1..L}, and r_l is obtained by
    masking *exactly* l response tokens chosen uniformly without replacement.

This is the LOW-VARIANCE, paper-recommended quality metric (Eq. 6), not the
single-random-t training loss. We average over `n_mc` Monte Carlo draws per
example (paper uses 128) to get a stable number.

Reported:
    - mean per-example conditional log-likelihood
    - mean per-token NLL  (= -LL / L)
    - corpus perplexity   (= exp(total NLL / total response tokens))

The prompt is never masked; only response tokens (query_mask == 1) are masked,
matching SFT training. Each example is scored on its OWN length (no batch
padding), so EOS-padding does not dilute the metric.
"""

import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from safetensors.torch import load_file
from tqdm import tqdm

from model import DeepfusionConfig, DeepfusionLM

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZER_PATH = os.path.join(_DIR, "DeepfusionLM_tokenizer")
DEFAULT_CHECKPOINT_PATH = os.path.join(_DIR, "my_experiment", "checkpoint_sft_model")
DEFAULT_DATA_PATH = os.path.join(_DIR, "data", "tokenized-sft")


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.pad_token = "[EOS]"

    config = DeepfusionConfig()
    config.vocab_size = len(tokenizer)
    model = DeepfusionLM(config)

    weights_file = (os.path.join(checkpoint_path, "model.safetensors")
                    if os.path.isdir(checkpoint_path) else checkpoint_path)
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Could not find weights at: {weights_file}")

    model.load_state_dict(load_file(weights_file), strict=True)
    print(f"Loaded weights from: {weights_file}")
    model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def conditional_log_likelihood(model, input_ids, response_positions, mask_token_id,
                               n_mc, mc_batch_size, device, generator):
    """
    Estimate log p_theta(response | prompt) for ONE example via Eq. (6),
    averaged over n_mc Monte Carlo draws. Returns (total_LL, response_len).
    """
    L = response_positions.numel()                 # response length
    seq_len = input_ids.numel()
    base = input_ids.to(device)                    # (seq_len,)
    response_positions = response_positions.to(device)
    true_tokens = base[response_positions]         # (L,) gold response tokens

    estimates = []                                 # per-MC-draw log-likelihood estimates

    remaining = n_mc
    while remaining > 0:
        b = min(mc_batch_size, remaining)
        remaining -= b

        ### Build b masked copies of this example ###
        batch = base.unsqueeze(0).repeat(b, 1)                      # (b, seq_len)
        labels = torch.full((b, seq_len), -100, dtype=torch.long, device=device)
        l_vec = torch.empty(b, device=device)

        for row in range(b):
            ### Step 3: l ~ Uniform{1..L} ###
            l = int(torch.randint(1, L + 1, (1,), generator=generator, device=device))
            l_vec[row] = l
            ### Step 4: mask exactly l response tokens, uniformly w/o replacement ###
            perm = torch.randperm(L, generator=generator, device=device)[:l]
            pos = response_positions[perm]
            batch[row, pos] = mask_token_id
            labels[row, pos] = true_tokens[perm]   # supervise only the masked response tokens

        ### Attend to everything (1 = valid), matching SFT training ###
        attn = torch.ones((b, seq_len), dtype=torch.long, device=device)
        logits = model(batch, attention_mask=attn)                 # (b, seq_len, vocab)

        ### Per-position NLL on masked tokens only (ignore_index=-100) ###
        vocab = logits.shape[-1]
        ce = F.cross_entropy(
            logits.reshape(b * seq_len, vocab).float(),
            labels.reshape(b * seq_len),
            reduction="none",
            ignore_index=-100,
        ).reshape(b, seq_len)
        ce_sum = ce.sum(dim=1)                                      # (b,) = -sum_i log p(masked)

        ### Eq. (6): estimate_i = (L / l) * sum log p = -(L / l) * ce_sum ###
        est = -(L / l_vec) * ce_sum                                # (b,) log-likelihood estimates
        estimates.append(est)

    total_ll = torch.cat(estimates).mean().item()                  # average over n_mc draws
    return total_ll, L


def main():
    parser = argparse.ArgumentParser("Conditional log-likelihood eval (LLaDA Algorithm 3 / Eq. 6)")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                        help="tokenized SFT dataset (must have input_ids + query_mask)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_mc", type=int, default=128,
                        help="Monte Carlo draws per example (paper uses 128)")
    parser.add_argument("--mc_batch_size", type=int, default=32,
                        help="how many MC copies to forward at once (lower if OOM)")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="evaluate only the first N examples (default: all)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path, args.tokenizer_path, args.device)
    mask_token_id = tokenizer.mask_token_id

    ### Lazy import so a broken pyarrow/datasets install can't block model loading ###
    from datasets import load_from_disk
    data = load_from_disk(args.data_path)
    ds = data[args.split] if hasattr(data, "keys") and args.split in data else data
    if args.num_examples is not None:
        ds = ds.select(range(min(args.num_examples, len(ds))))
    print(f"Evaluating {len(ds)} examples from split='{args.split}' with n_mc={args.n_mc}\n")

    ### Fixed generator for reproducibility ###
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    sum_ll = 0.0          # sum of per-example log-likelihoods
    sum_nll = 0.0         # sum of per-example NLL (= -LL)
    total_tokens = 0      # total response tokens (for corpus perplexity)
    n_scored = 0
    skipped = 0

    for ex in tqdm(ds, desc="Scoring"):
        input_ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        query_mask = torch.tensor(ex["query_mask"], dtype=torch.long)
        response_positions = torch.nonzero(query_mask == 1, as_tuple=False).squeeze(-1)

        if response_positions.numel() == 0:
            skipped += 1
            continue

        ll, L = conditional_log_likelihood(
            model, input_ids, response_positions, mask_token_id,
            args.n_mc, args.mc_batch_size, args.device, generator,
        )

        sum_ll += ll
        sum_nll += -ll
        total_tokens += L
        n_scored += 1

    if n_scored == 0:
        print("No examples scored (all had empty responses?).")
        return

    mean_ll = sum_ll / n_scored                       # mean per-example log-likelihood
    mean_token_nll = sum_nll / total_tokens           # token-weighted NLL
    corpus_ppl = math.exp(mean_token_nll)

    print("\n" + "=" * 56)
    print(f"  Examples scored          : {n_scored}  (skipped {skipped})")
    print(f"  Total response tokens    : {total_tokens}")
    print(f"  Monte Carlo draws (n_mc) : {args.n_mc}")
    print("-" * 56)
    print(f"  Mean per-example LL      : {mean_ll:10.4f}   (higher = better)")
    print(f"  Mean per-token NLL       : {mean_token_nll:10.4f}   (lower = better)")
    print(f"  Corpus perplexity        : {corpus_ppl:10.4f}   (lower = better)")
    print("=" * 56)
    print("\nNote: this is the low-variance Eq.(6) metric, not the training loss.")
    print("Perplexity ~ vocab size means near-random; single-digit is strong.")


if __name__ == "__main__":
    main()
