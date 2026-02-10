# Sprint 5 Innovation Priorities

## [REF:IP-S5] - Prioritized Innovation Backlog

**Status:** ACTIVE  
**Sprint Duration:** 2 weeks  
**Created:** January 2026  
**Branch:** `sprint6/api-integration` → `sprint5/kernel-optimization`

---

## Priority Matrix

| #   | Innovation                        | Impact                           | Effort | Priority | Week     |
| --- | --------------------------------- | -------------------------------- | ------ | -------- | -------- |
| 1   | **BitNet TL2_0 Parallel Kernels** | Critical (1.15-2.1x speedup)     | High   | P0       | Week 1   |
| 2   | **MRL Semantic Compression**      | High (4-64x embedding reduction) | Medium | P1       | Week 1-2 |
| 3   | **Binary Quantization Pipeline**  | Medium (memory reduction)        | Low    | P2       | Week 2   |
| 4   | **CI/CD Auto-Optimization**       | Medium (developer velocity)      | Medium | P2       | Week 2   |

---

## P0: BitNet TL2_0 Parallel Kernels (Week 1)

### Objective

Integrate January 2026 BitNet parallel kernel research into Ryzanstein's inference engine, achieving 1.15-2.1x speedup over current naive implementation.

### Technical Approach

1. **TL2_0 Element-wise LUT Method**: Pre-compute lookup tables for ternary {-1, 0, +1} × INT8 products
   - Weight w=-1 → result = -activation
   - Weight w=0 → result = 0
   - Weight w=+1 → result = +activation
   - Reduces multiply-accumulate to table lookup + accumulate

2. **Configurable Tiling**: Auto-tune tile sizes to CPU cache hierarchy
   - L1 tile (32KB): 32×32 blocks for register-level reuse
   - L2 tile (512KB): 128×128 blocks for L2 residency
   - L3 tile (32MB): Full row strips for streaming

3. **Multi-threaded GEMV**: OpenMP parallel execution
   - Row-parallel distribution across CPU cores
   - Dynamic scheduling for load balancing
   - Thread-local accumulators to avoid false sharing

4. **Embedding Quantization**: INT4/INT2 embedding compression
   - Reduce embedding table memory by 4-8×
   - Sub-byte packing with SIMD decompression

### Deliverables

- `src/core/bitnet/kernels/parallel_kernels.h` - TL2_0 header
- `src/core/bitnet/kernels/parallel_kernels.cpp` - TL2_0 implementation
- `src/core/bitnet/kernels/lut_gemm.h` - LUT-based GEMM header
- `src/core/bitnet/kernels/lut_gemm.cpp` - LUT-based GEMM implementation
- Unit tests in `src/core/bitnet/tests/`

### Success Criteria

- ≥1.5x speedup on GEMV (single-batch inference)
- Linear scaling to 8+ threads
- MSE < 1e-4 vs naive reference implementation
- Zero regressions on CI/CD

---

## P1: MRL Semantic Compression (Week 1-2)

### Objective

Implement Matryoshka Representation Learning (MRL) for multi-resolution embeddings with 4-64x compression.

### Technical Approach

1. Multi-resolution encoding: 2048 → 512 → 256 → 32 dimensions
2. Binary quantization: FP32 → 1-bit per dimension
3. CompresSAE integration for learned compression
4. Adaptive resolution selection based on query complexity

### Deliverables

- `src/inference/mrl_compression.py` - MRL pipeline
- Configuration for resolution tiers
- Benchmark suite

---

## P2: Binary Quantization Pipeline (Week 2)

### Objective

Enable 1-bit quantization for embedding vectors, reducing memory by 32x while maintaining retrieval quality.

---

## P2: CI/CD Auto-Optimization (Week 2)

### Objective

Enhance GitHub Actions workflow with CPU feature auto-detection, benchmark regression testing, and self-tuning compilation flags.

### Deliverables

- Enhanced `.github/workflows/ci.yml`
- `scripts/detect_cpu_features.py` - CPU topology detection
- `scripts/benchmark_regression.py` - Automated benchmark comparison
- Cache-aware compilation flags in CMakeLists.txt

---

## Dependencies

```
TL2_0 Parallel Kernels ─────┐
                             ├──→ CI/CD Auto-Optimization
MRL Semantic Compression ────┘

Binary Quantization ─────────→ MRL Compression (optional)
```

## Risk Assessment

| Risk                                 | Mitigation                                 |
| ------------------------------------ | ------------------------------------------ |
| AVX-512 not available on all targets | Runtime dispatch with scalar fallback      |
| LUT approach may not beat VNNI       | Benchmark both, keep best                  |
| Thread contention on small matrices  | Dynamic threshold for parallelism          |
| MRL quality loss too high            | Configurable resolution with quality floor |
