"""Model configuration for the from-scratch Llama implementation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LlamaConfig:
    """Configuration mirroring HuggingFace's LlamaConfig.

    We keep our own dataclass so the model code has zero dependency on
    `transformers` at runtime — the engine must be able to run standalone.
    """

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # GQA: < num_attention_heads for grouped query
    head_dim: int = 128            # usually hidden_size // num_attention_heads
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None  # e.g. Llama-3.2 "llama3" scaling
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False

    @classmethod
    def from_hf(cls, hf_config) -> "LlamaConfig":
        """Build our config from a HuggingFace LlamaConfig (drop-in).

        Handles both transformers 4.x (``rope_theta``/``rope_scaling`` as
        top-level fields) and 5.x (both folded into ``rope_parameters``).
        """
        rope_params = (
            getattr(hf_config, "rope_parameters", None)
            or getattr(hf_config, "rope_scaling", None)
            or {}
        )
        rope_theta = rope_params.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0))
        rope_type = rope_params.get("rope_type", "default")
        # Non-default rope_type (e.g. "llama3" scaled) is what we treat as scaling.
        rope_scaling = rope_params if rope_type != "default" else None

        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            head_dim=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            attention_bias=getattr(hf_config, "attention_bias", False),
            mlp_bias=getattr(hf_config, "mlp_bias", False),
        )
