"""From-scratch Llama architecture, weight-compatible with HuggingFace.

Module structure and parameter names mirror HF's ``LlamaForCausalLM`` exactly,
so a real checkpoint can be loaded with ``load_state_dict``. The forward pass
is our own — we control the attention, so we can later swap in paged attention
and custom CUDA kernels.

Current scope: full-sequence (causal) forward with plain RoPE + GQA + SwiGLU.
KV-cache, speculative decoding, and quantization are layered on next.
"""

from typing import Tuple

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
def precompute_rope(head_dim, seq_len, theta=10000.0, device=None):
    """Return (cos, sin) of shape (seq_len, head_dim) for positions 0..seq_len-1."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)          # (seq_len, head_dim // 2)
    emb = torch.cat([freqs, freqs], dim=-1)   # (seq_len, head_dim)
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


def causal_mask(seq, device, dtype):
    """Additive upper-triangular -inf mask: token i attends to positions <= i."""
    mask = torch.full((seq, seq), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


# ---------------------------------------------------------------------------
# Attention (GQA) — no cache yet
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

    def forward(self, x, cos, sin, mask):
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, k, cos, sin)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn + mask
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

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
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

    def forward(self, input_ids):
        bsz, seq = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = precompute_rope(self.config.head_dim, seq, self.config.rope_theta, input_ids.device)
        mask = causal_mask(seq, input_ids.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))
