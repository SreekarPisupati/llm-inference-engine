"""Autoregressive generation loop with a KV cache.

Prefill: run the whole prompt once, populate the cache.
Decode:  one token at a time, attending to the cache.

This is the baseline decode path. It is intentionally simple and slow
(one sequence, contiguous cache) — paged attention + continuous batching
replace it next, and every speedup is measured against this.
"""

from typing import Callable, Optional

import torch

from .kv_cache import KVCache
from .sampler import greedy


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    sampler: Optional[Callable] = None,
    eos_token_id: Optional[int] = None,
    max_seq: Optional[int] = None,
) -> torch.Tensor:
    """Greedy/sampled generation. Returns input_ids plus new tokens."""
    if sampler is None:
        sampler = greedy

    cfg = model.config
    device = input_ids.device
    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    if max_seq is None:
        max_seq = prompt_len + max_new_tokens

    cache = KVCache(
        num_layers=cfg.num_hidden_layers,
        batch=batch,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_seq=max_seq,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    generated = input_ids
    logits = model(input_ids, cache=cache)  # prefill

    for _ in range(max_new_tokens):
        next_token = sampler(logits[:, -1, :])          # (batch, 1)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break
        logits = model(next_token, cache=cache)         # decode one step

    return generated
