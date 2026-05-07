# c-transformer

Build a transformer-based language model end-to-end in C and CUDA — from BPE tokenizer through training loop to optimized inference kernels. A from-scratch implementation for understanding how modern LLMs work at every level of the stack.

## Prerequisites

- GCC or Clang (C99 / C11)
- CUDA Toolkit 12.x with nvcc
- NVIDIA GPU (consumer GPU for development; A100 for full training)
- Nsight Compute / Nsight Systems (optional, for profiling)

## Quick Start

```bash
git clone https://github.com/Pepps233/c-transformer.git
cd c-transformer
make
```

## Roadmap

### Phase 0: Foundations
CUDA hello world, Makefile, test harness, tensor struct with host/device transfer.

### Phase 1: BPE Tokenizer
Byte-pair encoding trained on TinyStories, with encode/decode round-trip verification.

### Phase 2: CPU Forward Pass
Full transformer architecture on CPU: embedding, layernorm, multi-head attention, GELU feedforward, residual connections, projection to logits.

### Phase 3: GPU Forward Pass
Port each operation to CUDA kernels. Verify against CPU outputs with tolerance 1e-4.

### Phase 4: Backward Pass and Training
Analytical gradients, AdamW optimizer, overfit-on-single-batch validation, full training loop.

### Phase 5: Real Training Run
Train ~5M parameter model on TinyStories, then scale to ~12M on A100s.

### Phase 6: Custom Kernels
Flash Attention, fused LayerNorm, fused GELU — benchmark against naive implementations.

### Phase 7: Polish
Inference binary, web demo, blog post with loss curves and kernel benchmarks.

See [ROADMAP.md](ROADMAP.md) for the full implementation plan drawn from the project design doc.

## Architecture

```
Token IDs → Embedding → [LayerNorm → Attention → Residual → LayerNorm → FFN → Residual] × N → Projection → Logits
```

GPU kernels compiled with nvcc and linked against host C code via Makefile. cuBLAS handles production matmul; custom kernels handle attention, layernorm, and GELU.

## References

Each paper is paired with its implementation phase:

- "Attention Is All You Need" (Vaswani et al., 2017) — Phase 2
- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016) — Phase 1
- "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019) — Phase 4
- "FlashAttention" (Dao et al., 2022) — Phase 6
- "Online Normalizer Calculation for Softmax" (Milakov & Gimelshein, 2018) — Phase 6

Karpathy's [llm.c](https://github.com/karpathy/llm.c) is used as a reference, not a template.