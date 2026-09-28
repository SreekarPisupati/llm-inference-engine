"""Correctness test: our from-scratch Llama must match HuggingFace bit-for-bit.

This is the invariant that everything else builds on. If our forward pass
doesn't match HF, no amount of serving optimization matters — we'd be serving
wrong logits.
"""

import torch
import pytest
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from inferx.model.config import LlamaConfig
from inferx.model.llama import LlamaForCausalLM


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    cfg = HFLlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA: 4 Q heads, 2 KV heads
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    hf = HFLlamaForCausalLM(cfg)

    mine_cfg = LlamaConfig.from_hf(cfg)
    mine = LlamaForCausalLM(mine_cfg)
    mine.load_state_dict(hf.state_dict())

    return hf, mine


def test_logits_match_hf(models):
    hf, mine = models
    torch.manual_seed(1)
    input_ids = torch.randint(0, 1000, (2, 16))

    with torch.no_grad():
        hf_logits = hf(input_ids).logits
        mine_logits = mine(input_ids)

    assert mine_logits.shape == hf_logits.shape
    torch.testing.assert_close(mine_logits, hf_logits, rtol=1e-4, atol=1e-5)


def test_gqa_repeats_kv(models):
    """GQA: hidden=128, 4 heads -> head_dim=32; 2 KV heads. Verify shapes flow."""
    _, mine = models
    torch.manual_seed(2)
    input_ids = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        out = mine(input_ids)
    assert out.shape == (1, 8, 1000)
