"""Token samplers: greedy, temperature, top-k, top-p (nucleus)."""

import torch


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Pick the argmax token. logits: (batch, vocab) -> (batch, 1)."""
    return logits.argmax(dim=-1, keepdim=True)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row. logits: (batch, vocab) -> (batch, 1)."""
    if temperature <= 0:
        return greedy(logits)

    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        top_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = top_vals[:, [-1]]
        logits = torch.where(logits < threshold, float("-inf"), logits)

    probs = torch.softmax(logits, dim=-1)

    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Remove tokens once the cumulative mass has already exceeded top_p.
        remove = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1, generator=generator)
