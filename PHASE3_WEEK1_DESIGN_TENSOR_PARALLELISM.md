# Task 1.1.2: Tensor Parallelism Strategy Design

**Document Type**: Architecture Design  
**Task ID**: 1.1.2  
**Assigned To**: @APEX  
**Status**: IN PROGRESS  
**Duration**: 2 days (Jan 1-2, 2026)  
**Priority**: CRITICAL

---

## Executive Summary

This document defines the tensor parallelism strategy for Ryzanstein LLM Phase 3, establishing the mathematical foundation and implementation approach for partitioning model weights across multiple GPUs. We adopt **row-wise tensor parallelism** as the primary strategy, optimizing for inference workloads with high compute-to-communication ratios.

---

## 1. Tensor Parallelism Strategy Overview

### 1.1 Strategy Selection: Row-Wise Partitioning

**Chosen Strategy**: Row-wise (input-feature) tensor parallelism  
**Rationale**: Optimal for inference where batch size is typically small (1-32) and sequence length is variable

#### Why Row-Wise Partitioning?

| Strategy          | Batch Inference | Latency | Communication | Best For          |
| ----------------- | --------------- | ------- | ------------- | ----------------- |
| **Row-wise**      | Excellent       | Low     | Minimal       | ✅ Our use case   |
| Column-wise       | Good            | Medium  | Moderate      | Training          |
| Sequence parallel | Moderate        | Medium  | High          | Long sequences    |
| Pipeline parallel | Fair            | High    | Very High     | Very large models |

**Key Advantage**: For inference with small batch sizes, row-wise partitioning minimizes communication overhead while maintaining high compute utilization.

---

## 2. Row-Wise Tensor Parallelism Implementation

### 2.1 Linear Layer Partitioning

#### Standard Linear Layer

```
Input: [batch_size, seq_len, hidden_dim]
Weight: [hidden_dim, output_dim]
Output: [batch_size, seq_len, output_dim]
```

#### Row-Wise Partitioned Linear Layer

```
GPU 0:  W_0 ∈ ℝ^(hidden_dim × output_dim/N)
GPU 1:  W_1 ∈ ℝ^(hidden_dim × output_dim/N)
...
GPU N:  W_N ∈ ℝ^(hidden_dim × output_dim/N)

Forward Pass:
  Y_i = X @ W_i    (each GPU computes partial output)
  Y = [Y_0 | Y_1 | ... | Y_N]  (concatenate results)

Backward Pass (training):
  dL/dX = [dL/dY_0 @ W_0^T | dL/dY_1 @ W_1^T | ...]
  all_reduce(dL/dX)  (gather gradients)
```

**Communication Cost**: One all-reduce per linear layer (gather gradients)  
**Computation per GPU**: 1/N of total computation  
**Memory per GPU**: 1/N of weight memory

### 2.2 Attention Layer Partitioning

#### Multi-Head Attention Parallelization

```
Standard MHA:
  Num Heads = H
  Head Dim = D

Partitioned MHA (row-wise by head):
  GPU 0: Heads [0 to H/N-1]
  GPU 1: Heads [H/N to 2H/N-1]
  ...
  GPU N: Heads [(N-1)H/N to H-1]

Forward Pass:
  1. Project QKV (row-wise linear layer)
  2. Compute attention per head partition (LOCAL)
  3. Concatenate head outputs: [head_0 | head_1 | ... | head_H-1]
  4. Output projection (row-wise linear layer)
  5. all_reduce() to gather final output
```

**Optimization**: Attention computation within each head partition is completely local (no communication), only projection layers require all-reduce.

**Communication Points**:

- Q/K/V projection: 1 all-reduce
- Output projection: 1 all-reduce

---

## 3. Communication Patterns

### 3.1 Forward Pass Communication

```
Linear Layer Forward (row-wise):
  Input: X (batch_size, seq_len, hidden_dim) [REPLICATED across GPUs]
  ├─ [GPU 0] Y_0 = X @ W_0  [output_dim/N features]
  ├─ [GPU 1] Y_1 = X @ W_1
  ├─ [GPU 2] Y_2 = X @ W_2
  └─ [GPU N] Y_N = X @ W_N

  Result: Partial outputs on each GPU, concatenate locally
  Communication: NONE (computation only)

Attention Output Projection:
  Projections complete on each GPU
  ├─ all_reduce(projected_outputs)  ← Gather results
  └─ [All GPUs] Full output

Communication: 1 all-reduce per attention layer
```

### 3.2 Backward Pass Communication (Training)

```
Backward through row-wise linear:
  dL/dY_i = dL/dY[partition_i]  [LOCAL]
  dL/dX_partial_i = dL/dY_i @ W_i^T

  ├─ [GPU 0] dL/dX_0 = dL/dY_0 @ W_0^T
  ├─ [GPU 1] dL/dX_1 = dL/dY_1 @ W_1^T
  └─ all_reduce(dL/dX)  ← Synchronize gradients

  Result: Full gradient available on all GPUs

Communication: 1 all-reduce per layer (backward)
```

---

## 4. Scaling Efficiency Analysis

### 4.1 Theoretical Scaling

For N GPUs, assuming perfect parallelization:

```
Speedup = N / (1 + Comm_Overhead)

Where:
  Comm_Overhead = T_communication / T_computation

For row-wise with small batches:
  T_comm ≈ 2 × (N-1)/N × all_reduce_latency
  T_comp ≈ Total_FLOPs / (GPU_Peak_FLOPs)

Typical values (V100, 8 GPUs):
  - All-reduce bandwidth: ~100 GB/s (NVLink)
  - All-reduce latency: <1ms
  - Inference compute: ~500 TFLOPS peak per GPU

  For 13B model with batch_size=4:
    T_comp ≈ 50ms
    T_comm ≈ 1ms per all_reduce
    Overhead ≈ 10/50 = 20%
    Efficiency ≈ 80% (target: >85%)
```

### 4.2 Empirical Scaling Targets

| GPU Count | Expected Speedup | Target Efficiency | Notes                      |
| --------- | ---------------- | ----------------- | -------------------------- |
| 1         | 1.0×             | 100%              | Baseline                   |
| 2         | 1.9×             | 95%               | Low communication overhead |
| 4         | **3.8× - 4.2×**  | **>95%**          | Sweet spot for inference   |
| 8         | 7.2× - 7.8×      | 90%+              | Good for larger batches    |

**Target**: 3.8-4.2× speedup on 4 GPUs (>95% efficiency)

---

## 5. Implementation Approach

### 5.1 PyTorch Abstractions

```python
# High-level API for row-wise tensor parallel

class RowParallelLinear(nn.Module):
    """
    Linear layer with row-wise tensor parallelism.

    Input:  [batch_size, seq_len, in_features]
    Output: [batch_size, seq_len, out_features]

    Weight distribution:
      out_features is partitioned across world_size GPUs
    """

    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert out_features % world_size == 0

        out_features_local = out_features // world_size
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features_local)
        )
        self.bias = nn.Parameter(torch.zeros(out_features_local))

    def forward(self, x):
        # x: [batch_size, seq_len, in_features]
        # Compute local partition
        output_local = F.linear(x, self.weight, self.bias)
        # output_local: [batch_size, seq_len, out_features_local]
        return output_local

class ColumnParallelLinear(nn.Module):
    """
    Column-wise parallel for backward pass synchronization.

    Input is replicated across GPUs.
    Output partitions are synchronized via all_reduce.
    """

    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert in_features % world_size == 0

        in_features_local = in_features // world_size
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features_local)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x partitioned: [batch_size, seq_len, in_features_local]
        output = F.linear(x, self.weight, self.bias)
        # All-reduce to gather outputs from all GPUs
        dist.all_reduce(output, op=ReduceOp.SUM)
        return output
```

### 5.2 Model Wrapper for Automatic Parallelization

```python
class DistributedModelWrapper(nn.Module):
    """
    Wraps a standard model and automatically parallelizes layers.
    """

    def __init__(self, model, world_size, rank):
        super().__init__()
        self.base_model = model
        self.world_size = world_size
        self.rank = rank

        # Replace linear layers
        self._parallelize_layers()

    def _parallelize_layers(self):
        """Replace nn.Linear with parallel versions."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Determine which type of parallelization
                parent = self._get_parent(self.base_model, name)

                if self._is_output_projection(name):
                    parallel_linear = RowParallelLinear(
                        module.in_features,
                        module.out_features,
                        self.world_size,
                        self.rank
                    )
                else:
                    parallel_linear = ColumnParallelLinear(
                        module.in_features,
                        module.out_features,
                        self.world_size,
                        self.rank
                    )

                setattr(parent, name.split('.')[-1], parallel_linear)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
```

---

## 6. Key Design Decisions

### Decision 1: Row-wise Over Column-wise for Inference

**Rationale**: Small batch sizes typical in inference make row-wise optimal due to lower communication overhead.

### Decision 2: All-reduce for Synchronization

**Rationale**: All-reduce is well-optimized in NCCL, provides implicit gradient synchronization, and is proven in production systems.

### Decision 3: Local Attention Computation

**Rationale**: Head-wise partitioning allows attention computation to remain local, minimizing communication in the most compute-intensive layer.

### Decision 4: Explicit vs. Implicit Parallelization

**Recommendation**: Explicit model wrapper (vs. fully automatic) provides clarity, debuggability, and allows per-layer optimization decisions.

---

## 7. Performance Benchmarking Plan

### 7.1 Micro-Benchmarks

```python
# Benchmark all_reduce latency
def benchmark_all_reduce(tensor_size_mb, world_size):
    """Measure all-reduce latency for various tensor sizes."""
    tensor = torch.randn(tensor_size_mb * 1024 * 1024 // 4)

    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor)

    # Measure
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / 100 * 1000  # ms per all-reduce
```

### 7.2 Model-Level Benchmarks

```
Metric: Forward pass latency, memory usage, throughput
Models: 7B, 13B, 33B parameter sizes
Batch sizes: 1, 4, 8, 16
GPU counts: 1, 2, 4, 8
Expected: >95% efficiency on 4 GPUs
```

---

## 8. Validation Criteria

- [ ] All-reduce latency <1ms per layer (measured)
- [ ] Computation vs. communication ratio >50:1
- [ ] 4-GPU scaling efficiency >95%
- [ ] Memory per GPU within budget (model_size/4 + activation overhead)
- [ ] Attention layer local computation verified
- [ ] Model wrapper successfully parallelizes all linear layers
- [ ] Correctness validation against single-GPU baseline

---

## 9. Next Steps

1. **Task 1.1.3**: Design multi-GPU orchestrator (depends on this design)
2. **Task 1.1.4**: Select NCCL communication backend
3. **Task 1.1.5**: Implement tensor parallelism layer (implementation phase)

---

## References

- Ring all-reduce algorithm (Hung et al., 2019)
- Megatron-LM tensor parallelism (Shoeybi et al., 2019)
- PyTorch distributed documentation
- NCCL optimization guides
