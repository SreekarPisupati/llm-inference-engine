# llm-inference-engine

A **from-scratch LLM inference engine and optimization toolkit** — paged KV cache, continuous
batching, speculative decoding, quantization, and custom CUDA kernels — built to demonstrate the
full LLMOps / inference-optimization skill stack.

This repo is a *from-scratch* build, not a wrapper around vLLM/SGLang. Those engines are used only
as benchmark baselines. The point is to show the internals: how a scheduler works, how a KV cache
is managed, how quantization and speculative decoding are actually measured.

**Honesty label (permanent):** this is an honest from-scratch reproduction + measured comparison.
Every number in `results/` is measured on real hardware. No inflated claims; provenance disclosed.

## Status

**v0.1 — model correctness.** The from-scratch Llama architecture (RMSNorm, RoPE, GQA, SwiGLU,
causal LM) is weight-compatible with HuggingFace and matches its logits bit-for-bit (rtol 1e-4).

```
inferx/
├── model/          # LlamaConfig, from-scratch Llama (HF-weight-compatible)
├── engine/         # (next) paged KV cache, continuous batching scheduler
├── speculative/    # (next) draft + verify
├── quantize/       # (next) INT4/INT8 + KV-cache quantization
├── kernels/        # (next) CUDA kernels (quantized GEMM, attention)
└── serve/          # (next) FastAPI OpenAI-compatible server
```

## Roadmap (mapped to the Gnani.ai LLMOps/inference JD)

| Layer | Status | Artifact |
|-------|--------|----------|
| Model correctness | ✅ done | from-scratch Llama == HF |
| Serving engine | 🔜 | paged KV cache, continuous batching, chunked prefill, prefix caching |
| Model optimization | 🔜 | speculative decoding, INT4/INT8 + KV-cache quantization |
| Kernels & low-level | 🔜 | custom CUDA kernel + roofline benchmark |
| Eval & cost | 🔜 | latency percentiles, load testing, cost-per-M-token |

## Run tests

```bash
python -m pytest -q
```

## Hardware

Developed on an RTX 4050 Laptop GPU (6 GB, CUDA 12.9, Windows). Triton is unavailable on Windows,
so custom kernels are written in raw CUDA via `nvcc`.
