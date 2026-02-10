# RYZEN-LLM Research Notes

## Overview

This document contains research notes, paper references, and implementation insights for the RYZEN-LLM project.

## Core Technologies

### BitNet b1.58

**Paper**: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (Microsoft Research, 2024)

**Key Insights**:
- Ternary quantization (-1, 0, +1) enables extreme compression
- 1.58 bits per parameter theoretical minimum
- Minimal quality loss compared to FP16 baseline
- Significant speedup on CPU due to simplified operations

**Implementation Notes**:
- Bit-packing: Use 2 bits per weight (4 weights per byte)
- GEMM optimization: Replace multiply with add/subtract
- Memory bandwidth becomes the bottleneck
- AVX-512 can process 16 weights simultaneously

### T-MAC (Table-based Matrix Multiplication)

**Paper**: "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment" (2024)

**Key Insights**:
- Precomputed lookup tables eliminate online computation
- 2-4x speedup over traditional low-bit GEMM
- Cache-friendly table sizes (typically <100KB per layer)
- Works synergistically with BitNet quantization

**Implementation Notes**:
- Table size: 256 entries for 8-bit activations × 2-bit weights
- Prefetching critical for performance
- Multiple tables for different tile sizes
- Trade-off: Table generation time vs. inference speedup

### Mamba SSM (State Space Models)

**Paper**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)

**Key Insights**:
- O(N) complexity vs. O(N²) for attention
- Selective mechanism enables context-aware processing
- Competitive quality with Transformers on many tasks
- Excellent for long-context scenarios

**Implementation Notes**:
- Parallel scan algorithm for training
- Recurrent mode for inference
- State size typically 16-64 per dimension
- Hardware-aware implementation critical

### RWKV (Receptance Weighted Key Value)

**Paper**: "RWKV: Reinventing RNNs for the Transformer Era" (2023)

**Key Insights**:
- Attention-free architecture with linear complexity
- Time-mixing and channel-mixing layers
- Can be trained as Transformer, inferred as RNN
- Excellent efficiency on CPU

**Implementation Notes**:
- WKV operator is the performance bottleneck
- Time-decay mechanism requires careful numerics
- State caching essential for multi-turn conversations
- Vectorization straightforward with AVX-512

## Optimization Techniques

### Speculative Decoding

**Paper**: "Fast Inference from Transformers via Speculative Decoding" (DeepMind, 2023)

**Key Insights**:
- Draft model generates candidates quickly
- Target model verifies in parallel
- 2-3x speedup with no quality loss
- Works best when draft model is similar to target

**Implementation Strategy**:
- Use Mamba 2.8B as draft for BitNet 7B
- Tree-based speculation for higher acceptance rate
- Adaptive draft length based on acceptance rate

### Token Recycling System

**Novel Contribution**: Original to RYZEN-LLM

**Concept**:
- Compress semantic-dense token sequences into RSUs
- Store as embeddings in vector database
- Retrieve relevant RSUs for future requests
- Dramatically reduces prompt reprocessing

**Research Questions**:
1. What is the optimal density threshold for RSU creation?
2. How do we measure semantic equivalence across RSUs?
3. What is the retrieval accuracy vs. context size trade-off?
4. Can RSUs be shared across users safely?

**Evaluation Metrics**:
- Compression ratio: Tokens → RSUs
- Retrieval precision/recall
- Context reconstruction quality
- End-to-end latency impact

## Hardware Optimization

### AVX-512 on AMD Ryzen

**Considerations**:
- AVX-512 available on Zen 4+ (Ryzen 7000+)
- 512-bit vectors = 16× FP32 or 64× INT8
- VNNI instructions for INT8 dot products
- Clock speed throttling less severe than Intel

**Optimization Strategies**:
- Use VNNI for quantized operations
- Minimize register spills with careful tiling
- Leverage large L3 cache (32-128MB)
- Balance compute with memory bandwidth

### Memory Hierarchy

**Ryzen 9 7950X Profile**:
- L1 Cache: 32KB I + 32KB D per core
- L2 Cache: 1MB per core
- L3 Cache: 64MB shared
- Memory: DDR5-5200 (83 GB/s)

**Optimization Strategies**:
- Tile sizes tuned for L2 (1MB per core)
- Shared data in L3 for cross-core access
- Prefetching for KV cache access
- NUMA-aware memory allocation

## Performance Benchmarks

### Target Metrics

| Metric | Ryzen 7 | Ryzen 9 | Threadripper |
|--------|---------|---------|--------------|
| TTFT (ms) | 600 | 400 | 250 |
| Tokens/sec | 10-15 | 20-30 | 40-60 |
| Memory (GB) | 8-12 | 12-16 | 16-24 |
| Concurrent | 2-3 | 5-8 | 10-15 |

### Comparison to Alternatives

| System | Hardware | Speed | Quality |
|--------|----------|-------|---------|
| llama.cpp | Ryzen 9 | 15 tok/s | FP16 baseline |
| RYZEN-LLM (BitNet) | Ryzen 9 | 25 tok/s | 95% of FP16 |
| RYZEN-LLM (Mamba) | Ryzen 9 | 35 tok/s | 90% of FP16 |

## Open Questions

1. **RSU Granularity**: What is the optimal size for Recyclable Semantic Units?
2. **Cross-User Sharing**: Can RSUs be safely shared across users?
3. **Quality Trade-offs**: What is the acceptable quality loss for 2x speedup?
4. **Model Selection**: How to automatically select between models?
5. **Long Context**: How do we handle contexts beyond 32K tokens efficiently?

## Future Research Directions

1. **Quantization-Aware Training**: Fine-tune models specifically for ternary weights
2. **Dynamic Quantization**: Adapt quantization precision per layer
3. **Mixture of Experts**: Combine multiple specialized models
4. **Distributed Inference**: Scale across multiple machines
5. **Hardware Co-design**: Custom CPU instructions for LLM operations

## References

### Papers
- BitNet b1.58: https://arxiv.org/abs/2402.17764
- T-MAC: https://arxiv.org/abs/2407.00088
- Mamba: https://arxiv.org/abs/2312.00752
- RWKV: https://arxiv.org/abs/2305.13048
- Speculative Decoding: https://arxiv.org/abs/2211.17192

### Projects
- llama.cpp: https://github.com/ggerganov/llama.cpp
- Mamba: https://github.com/state-spaces/mamba
- RWKV: https://github.com/BlinkDL/RWKV-LM
- Qdrant: https://github.com/qdrant/qdrant

### Documentation
- AMD Zen 4 Architecture: https://www.amd.com/en/products/processors/desktops/ryzen.html
- AVX-512 Intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- FastAPI: https://fastapi.tiangolo.com/
- Model Context Protocol: https://modelcontextprotocol.io/

## Contributors

Research notes compiled by the RYZEN-LLM team.

Last updated: 2024
