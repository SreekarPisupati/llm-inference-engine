"""From-scratch Llama architecture, weight-compatible with HuggingFace.

Module structure and parameter names mirror HF's ``LlamaForCausalLM`` exactly,
so a real checkpoint can be loaded with ``load_state_dict``. The forward pass is
our own — we control the attention, so we can swap in paged attention and
custom CUDA kernels later.

Scope now: causal forward with plain RoPE + GQA + SwiGLU, plus an optional
KV cache for incremental (decode-step) inference.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Llama-style).

    Casts to float32 for the variance computation (matching HF) so results are
    bit-comparable in fp32/bf16.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


# ---------------------------------------------------------------------------
# Rotary position embeddings (RoPE)
# ---------------------------------------------------------------------------
def rope_cos_sin(head_dim, positions, theta=10000.0, device=None):
    """Return (cos, sin) of shape (len(positions), head_dim) for those positions."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)  # (n_pos, head_dim // 2)
    emb = torch.cat([freqs, freqs], dim=-1)                     # (n_pos, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    """Apply RoPE to q,k of shape (bsz, n_heads, seq, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


def causal_mask(seq, total, past_len, device, dtype):
    """Additive mask of shape (seq, total).

    Query i (absolute position past_len + i) may attend to keys 0..past_len+i.
    Handles prefill (past_len=0), decode (seq=1), and chunked prefill generally.
    """
    q_pos = torch.arange(past_len, past_len + seq, device=device).unsqueeze(1)  # (seq, 1)
    k_pos = torch.arange(total, device=device).unsqueeze(0)                     # (1, total)
    allowed = k_pos <= q_pos
    return torch.zeros((seq, total), device=device, dtype=dtype).masked_fill(~allowed, float("-inf"))


# ---------------------------------------------------------------------------
# Attention (GQA) with optional KV cache
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(self, x, cos, sin, cache=None, layer_idx=0, past_len=0):
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, k, cos, sin)

        if cache is not None:
            cache.write(layer_idx, k, v)
            total = past_len + seq
            k, v = cache.read(layer_idx, total)
        else:
            total = seq

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn + causal_mask(seq, total, past_len, x.device, attn.dtype)
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.o_proj(out)

    def _repeat_kv(self, x):
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep == 1:
            return x
        # (bsz, n_kv, seq, head_dim) -> (bsz, n_heads, seq, head_dim)
        return x[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(
            x.shape[0], self.num_heads, x.shape[2], self.head_dim
        )


# ---------------------------------------------------------------------------
# MLP (SwiGLU)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, cache=None, layer_idx=0, past_len=0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx, past_len)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# Model + causal LM head
# ---------------------------------------------------------------------------
class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, cache=None, position_ids=None):
        bsz, seq = input_ids.shape
        past = cache.len if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past, past + seq, device=input_ids.device)

        cos, sin = rope_cos_sin(
            self.config.head_dim, position_ids, self.config.rope_theta, input_ids.device
        )
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache, i, past)
        x = self.norm(x)

        if cache is not None:
            cache.advance(seq)
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, cache=None, position_ids=None):
        hidden = self.model(input_ids, cache=cache, position_ids=position_ids)
        return self.lm_head(hidden)
