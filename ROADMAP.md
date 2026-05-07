# Building a Transformer From Scratch in C — Full Roadmap

A phase-by-phase implementation plan for building a small transformer-based language model end-to-end in C and CUDA. Reference: Karpathy's llm.c.

## Recommended Stack

| Component | Choice |
|-----------|--------|
| Language | C99 / C11 (GCC or Clang) |
| GPU | CUDA Toolkit 12.x with nvcc |
| Build | Plain Makefile |
| Profiling | Nsight Compute (ncu) + Nsight Systems (nsys) |
| Math | cuBLAS for production matmul; naive matmul for learning |
| Data | Flat binary of uint16_t token IDs, memory-mapped |
| RNG | Mersenne Twister or PCG (not rand()) |
| Testing | Minimal custom harness (~50 lines) or µnit |
| Dataset | TinyStories (~2 GB plain text) |
| Compute | Local GPU for dev; Modal/Lambda Labs A100 for full training |

## Phase 0: Foundations

Objective: confirm the toolchain works end-to-end before any ML code.

Deliverables:
- Working CUDA installation
- "Hello world" kernel (vector add), profiled with ncu
- Makefile that compiles and links C and CUDA together
- Basic test harness
- Tensor struct: `{ float* data; int* shape; int ndim; size_t numel; bool on_gpu; }` with alloc/free/host-device transfer functions

## Phase 1: BPE Tokenizer in C

- Read Sennrich et al. (2016) on subword units
- Implement BPE training: count token pairs, merge most frequent, repeat
- Data structures: open-addressing hashmap (~200 lines) or uthash
- Verify on small corpus first — early merges should produce "th", "he", "in"
- Train on full TinyStories
- Outputs: vocabulary file, merges file, encoder
- Validation: encode-then-decode round-trip on arbitrary strings

## Phase 2: CPU Forward Pass

Implement full transformer architecture on CPU with naive C loops:
- Embedding lookup
- LayerNorm
- Multi-head attention (single-head first, then generalize)
- Feedforward block with GELU activation
- Residual connections
- Final projection to vocabulary logits

Verification: tiny model (2 layers, 4 heads, dim 32) with hard-coded weights must match hand-computed values on 5-token input.

## Phase 3: GPU Forward Pass

Port Phase 2 to CUDA one operation at a time:
1. Elementwise ops (add, GELU) — validate GPU tensor plumbing
2. LayerNorm
3. Matmul (naive, then cuBLAS)
4. Attention

After each kernel: test CPU vs GPU with identical inputs, verify within 1e-4 tolerance.

## Phase 4: Backward Pass and Training Loop

- Derive each gradient on paper before coding
- Matmul backward: two matmuls
- LayerNorm backward: chain rule through normalization (subtle)
- Attention backward: softmax derivative
- Gradient checking on every operation: `(f(x+ε) − f(x−ε)) / (2ε)` vs analytical
- AdamW optimizer (Loshchilov & Hutter, 2019) — decoupled weight decay
- Training loop: batch load → forward → loss → backward → optimizer step
- First milestone: overfit on single batch of ~1000 tokens

## Phase 5: Real Training Run

- Small scale: ~4 layers, 256 hidden, 4 heads, ~5M params on consumer GPU
- Verify: loss decreases, output becomes story-like
- Scale up: ~12M params on A100 (Modal / Lambda Labs, ~tens of dollars)
- Standard practices: log loss, checkpoint regularly, write inference code

## Phase 6: Custom Optimized Kernels

- Flash Attention (Dao et al., 2022) — forward then backward
- Fused LayerNorm — single kernel, no intermediate HBM writes
- Fused GELU into feedforward matmul (optional)
- Benchmark every kernel against naive counterpart with Nsight Compute
- Reference: Online Normalizer Calculation for Softmax (Milakov & Gimelshein, 2018)

## Phase 7: Polish and Showcase

- Clean inference binary: load checkpoint, generate from prompt
- Simple web demo (subprocess wrapper around inference binary)
- Blog post: architecture, kernel benchmarks with measured numbers, loss curves
- Polished README

## Practical Warnings

- Use FP32 everywhere initially; mixed precision belongs in Phase 6
- Target single GPU architecture (e.g., sm_86 for RTX 30 series)
- Every cudaMalloc needs a corresponding cudaFree — arena allocator recommended
- Use llm.c as a reference, not a template — diff against it to localize bugs

## Reading List

Each paper is most useful when paired with its implementation phase:

| Paper | Phase | When |
|-------|-------|------|
| "Attention Is All You Need" (Vaswani et al., 2017) | Phase 2 | Read fully before implementing architecture |
| "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016) | Phase 1 | Short, self-contained; read before BPE |
| "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019) | Phase 4 | Read when implementing AdamW |
| "FlashAttention" (Dao et al., 2022) | Phase 6 | Read after experiencing naive attention cost |
| "Online Normalizer Calculation for Softmax" (Milakov & Gimelshein, 2018) | Phase 6 | Read alongside FlashAttention paper |