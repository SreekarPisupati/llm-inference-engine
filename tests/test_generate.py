"""Decode-loop correctness: our KV-cache generation must match HF's generate().

The forward pass already matches HF (test_correctness). This test proves the
*incremental* path — prefill + cached decode — produces the same tokens as HF's
own optimized generation. If this diverges, the cache is wrong.
"""

import torch
import pytest
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from inferx.model.config import LlamaConfig
from inferx.model.llama import LlamaForCausalLM
from inferx.engine.generate import generate
from inferx.engine.sampler import greedy


@pytest.fixture(scope="module")
def tiny_models():
    torch.manual_seed(0)
    cfg = HFLlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    hf = HFLlamaForCausalLM(cfg)
    mine = LlamaForCausalLM(LlamaConfig.from_hf(cfg))
    mine.load_state_dict(hf.state_dict())
    return hf, mine


def test_greedy_generation_matches_hf(tiny_models):
    hf, mine = tiny_models
    torch.manual_seed(1)
    input_ids = torch.randint(0, 1000, (1, 8))

    with torch.no_grad():
        ours = generate(mine, input_ids, max_new_tokens=12, sampler=greedy)
        theirs = hf.generate(input_ids, max_new_tokens=12, do_sample=False)

    assert ours.shape == theirs.shape
    assert torch.equal(ours, theirs), f"\nours:   {ours}\ntheirs: {theirs}"


def test_cache_reuse_does_not_change_logits(tiny_models):
    """A single cached decode step must equal a full-sequence forward."""
    _, mine = tiny_models
    torch.manual_seed(2)
    prompt = torch.randint(0, 1000, (1, 10))

    with torch.no_grad():
        full = mine(prompt)                       # no cache: full-sequence forward
        step = mine(prompt[:, :9], cache=None)    # first 9
        # now cache the first 9 and decode the 10th
        from inferx.engine.kv_cache import KVCache
        cfg = mine.config
        cache = KVCache(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, cfg.head_dim,
                        max_seq=16, device=prompt.device, dtype=torch.float32)
        mine(prompt[:, :9], cache=cache)
        cached = mine(prompt[:, 9:10], cache=cache)

    torch.testing.assert_close(cached[:, -1, :], full[:, -1, :], rtol=1e-4, atol=1e-5)
