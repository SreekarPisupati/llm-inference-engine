"""Naive contiguous KV cache — the baseline that paged attention will replace.

This reserves a full ``(layers, batch, num_kv_heads, max_seq, head_dim)`` buffer
and hands each request a contiguous slice. It is exactly the memory-wasteful
design the vLLM paper calls out:

  * space is reserved up to ``max_seq`` even though most requests are far shorter
  * the reservation cannot be shared across requests that share a prefix
  * fragmentation means we fit fewer concurrent requests than VRAM allows

Keeping it here as the *baseline* is the point: every later optimization gets
benchmarked against this, and the memory numbers prove why paging wins.
"""

import torch


class KVCache:
    def __init__(self, num_layers, batch, num_kv_heads, head_dim, max_seq, device, dtype):
        self.k = torch.zeros(
            num_layers, batch, num_kv_heads, max_seq, head_dim, device=device, dtype=dtype
        )
        self.v = torch.zeros_like(self.k)
        self.len = 0          # number of tokens currently cached
        self.max_seq = max_seq

    def write(self, layer_idx, k, v):
        """Write new k/v at the current end. Does NOT advance the length.

        Called once per layer inside a single forward pass; advancing happens
        once, after every layer has written (see :meth:`advance`).
        """
        seq = k.shape[2]
        end = self.len + seq
        if end > self.max_seq:
            raise ValueError(f"KV cache overflow: {end} > {self.max_seq}")
        self.k[layer_idx, :, :, self.len:end, :] = k
        self.v[layer_idx, :, :, self.len:end, :] = v

    def read(self, layer_idx, total):
        """Return cached k/v up to ``total`` tokens (for attention)."""
        return (
            self.k[layer_idx, :, :, :total, :],
            self.v[layer_idx, :, :, :total, :],
        )

    def advance(self, seq):
        self.len += seq

    def memory_bytes(self):
        return self.k.numel() * self.k.element_size() * 2
