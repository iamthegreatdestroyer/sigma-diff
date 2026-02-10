# Task 1.1.4: NCCL Communication Backend Selection & Configuration

**Document Type**: Architecture Design  
**Task ID**: 1.1.4  
**Assigned To**: @APEX  
**Status**: IN PROGRESS  
**Duration**: Concurrent with 1.1.1-1.1.3  
**Priority**: CRITICAL

---

## Executive Summary

This document defines the selection and configuration of NCCL (NVIDIA Collective Communications Library) as the communication backend for Ryzanstein LLM's distributed inference. NCCL is the industry standard for GPU collective operations, providing highly optimized all-reduce, broadcast, and other operations critical for tensor parallelism.

---

## 1. Backend Selection Rationale

### 1.1 Communication Backend Comparison

| Criteria                  | NCCL    | Gloo     | MPI       | Custom   |
| ------------------------- | ------- | -------- | --------- | -------- |
| **GPU Collective Ops**    | ⭐⭐⭐  | ⭐       | ⭐⭐      | ❌       |
| **Bandwidth Utilization** | 95%+    | 60-70%   | 85%+      | Variable |
| **Latency (all-reduce)**  | <1ms    | 10-50ms  | 5-20ms    | Unknown  |
| **NVLink Optimization**   | ⭐⭐⭐  | ⭐       | ⭐        | ❌       |
| **Production Ready**      | ⭐⭐⭐  | ⭐⭐     | ⭐⭐⭐    | ❌       |
| **Maintained**            | Active  | Active   | Active    | Varies   |
| **Our Use Case**          | ✅ Best | Fallback | Not ideal | ❌       |

**Selection**: **NCCL 2.18+** (latest stable) for GPU-based distributed inference

**Fallback**: **Gloo** for CPU-only scenarios or development without GPUs

---

## 2. NCCL Architecture & Capabilities

### 2.1 NCCL Collective Operations

```
┌─────────────────────────────────────────────────────────────┐
│                   NCCL OPERATIONS                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ALL_REDUCE     [1,2,3] + [4,5,6] + [7,8,9] = [12,15,18]  │
│  ├─ Compute operation: SUM, MAX, MIN, PROD                 │
│  ├─ Use case: Gradient synchronization, metric aggregation │
│  └─ Latency: O(log N) with tree reduction                  │
│                                                              │
│  BROADCAST      [1,2,3] from rank 0 → all ranks            │
│  ├─ Use case: Model weights, hyperparameters               │
│  ├─ Latency: O(log N) with tree propagation                │
│  └─ Optimization: NVLink direct copy for root              │
│                                                              │
│  ALL_GATHER     [1,2] [3,4] [5,6] [7,8] → [1,2,3,4,5,6,7,8]
│  ├─ Use case: Collecting outputs from all GPUs             │
│  ├─ Latency: O(log N) with optimized gather algorithm      │
│  └─ Memory overhead: N copies of data per GPU              │
│                                                              │
│  REDUCE_SCATTER [1,2,3,4,5,6,7,8] → [1] [2] [3] [4]       │
│  ├─ Use case: Distributing gradients                       │
│  ├─ Latency: O(log N)                                      │
│  └─ Optimal for: Distributed gradient aggregation          │
│                                                              │
│  SEND/RECV      Rank i → Rank j (point-to-point)           │
│  ├─ Use case: Pipelined inference                          │
│  ├─ Latency: Direct GPU-to-GPU                             │
│  └─ Bandwidth: Full NVLink bandwidth                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 NCCL Communication Patterns

For 4 GPUs with ring all-reduce:

```
Ring All-Reduce Algorithm (Most Bandwidth Efficient)

Step 1: GPU i sends slice i to GPU (i+1) mod N
  GPU 0 → GPU 1: Slice [0]      GPU 2 → GPU 3: Slice [2]
  GPU 1 → GPU 2: Slice [1]      GPU 3 → GPU 0: Slice [3]

Step 2-3: Propagate and accumulate
  Each GPU receives, accumulates, forwards

Total bandwidth: 2(N-1)/N ≈ 1.5x for N=4
Latency: O(N) hops of latency

Tree All-Reduce (Lower Latency)
  Step 1: Reduce tree (bottom-up)
  Step 2: Broadcast tree (top-down)
  Latency: O(log N), Bandwidth: Lower
```

---

## 3. NCCL Configuration for Ryzanstein

### 3.1 Environment Variables

**Critical NCCL Environment Variables:**

```bash
# Debugging and Logging
export NCCL_DEBUG=INFO              # Verbose logging (TRACE, DEBUG, INFO, WARN)
export NCCL_DEBUG_SUBSYS=ALL        # All subsystems
export NCCL_DEBUG_LOG_DIR=./logs    # Log file directory

# Network Configuration
export NCCL_SOCKET_IFNAME=eth0      # Network interface (auto-detected normally)
export NCCL_IB_DISABLE=1            # Disable InfiniBand (if not available)

# Algorithm Selection
export NCCL_ALGO=Ring               # Ring for bandwidth, Tree for latency
export NCCL_PROTO=Simple             # Protocol version

# Timing and Synchronization
export NCCL_TIMEOUT=600              # Timeout in seconds (600s = 10min)
export NCCL_LAUNCH_MODE=PARALLEL    # Parallel vs sequential launch

# GPU Direct RDMA (for DGX/multi-socket)
export NCCL_P2P_LEVEL=PCI           # P2P access level: LOC, SYS, PCI, PIX

# Performance Optimization
export NCCL_MIN_NRINGS=4            # Minimum number of rings
export NCCL_MAX_NRINGS=16           # Maximum number of rings
export NCCL_BUFFSIZE=16777216       # Communication buffer size (16MB)

# NVLink Optimization
export NCCL_NVLINK_DISABLE=0        # Enable NVLink if available
export NCCL_NET_GDR_LEVEL=3         # GPU Direct RDMA level for network
```

### 3.2 Recommended Configuration

**For Ryzanstein Phase 3 (4 GPUs with NVLink):**

```bash
# Optimal settings for inference workload
export NCCL_DEBUG=WARN              # Warnings only (not verbose)
export NCCL_ALGO=Ring               # Ring for throughput
export NCCL_PROTO=LL                # Low-latency protocol
export NCCL_MAX_NRINGS=8            # Multiple rings for parallelism
export NCCL_TIMEOUT=600
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_P2P_LEVEL=PCI

# For debugging (development only)
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_LOG_DIR=./nccl_logs
```

### 3.3 PyTorch NCCL Initialization

```python
import torch
import torch.distributed as dist
from datetime import timedelta

def init_nccl_backend(rank: int, world_size: int,
                     master_addr: str = "localhost",
                     master_port: int = 29500,
                     timeout_minutes: int = 30):
    """
    Initialize PyTorch with NCCL backend for GPU communication.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        master_addr: Master process address
        master_port: Master process port
        timeout_minutes: Timeout for collective operations

    Raises:
        RuntimeError: If initialization fails
    """

    # Set environment variables programmatically
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    # Initialize process group with NCCL backend
    try:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=timeout_minutes)
        )

        # Verify NCCL version
        print(f"NCCL Version: {torch.cuda.nccl.version()}")

        # Test basic all-reduce
        test_tensor = torch.ones(1, device=f"cuda:{rank}")
        dist.all_reduce(test_tensor)

        if test_tensor.item() != world_size:
            raise RuntimeError("NCCL all-reduce verification failed")

        print(f"Rank {rank}: NCCL backend initialized successfully")

    except Exception as e:
        print(f"Rank {rank}: NCCL initialization failed: {e}")
        raise

def cleanup_nccl():
    """Cleanup NCCL resources."""
    if dist.is_initialized():
        dist.destroy_process_group()
```

---

## 4. NCCL Performance Optimization

### 4.1 All-Reduce Optimization for Row-Wise Parallelism

```python
class OptimizedAllReduce:
    """
    Optimized all-reduce for our tensor parallelism pattern.
    """

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.accumulated_bytes = 0
        self.num_operations = 0

    def efficient_all_reduce(self, tensor: torch.Tensor,
                            op: str = "sum") -> torch.Tensor:
        """
        Perform all-reduce with profiling and optimization.

        Optimizations:
        1. Gradient accumulation (batch all-reduces if possible)
        2. Mixed precision (all-reduce in lower precision)
        3. Asynchronous all-reduce where safe
        """

        # Profile
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Perform all-reduce
        if op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Track metrics
        self.accumulated_bytes += tensor.numel() * tensor.itemsize
        self.num_operations += 1

        # Log if slow
        if elapsed > 0.01:  # More than 10ms
            print(f"Rank {self.rank}: Slow all-reduce "
                  f"({tensor.numel()} elements, {elapsed*1000:.2f}ms)")

        return tensor

    def get_bandwidth_utilized(self) -> float:
        """Estimate bandwidth utilization."""
        if self.num_operations == 0:
            return 0.0

        # Rough estimate (not exact due to protocol overhead)
        avg_op_time = self.accumulated_bytes / self.num_operations / 1e9
        nvlink_bandwidth = 100  # GB/s for NVLink

        return min(100.0, (self.accumulated_bytes / 1e9 / avg_op_time) / nvlink_bandwidth * 100)
```

### 4.2 Bandwidth Estimation

```
NCCL All-Reduce Bandwidth Analysis:

Ring Algorithm (ideal for 4 GPUs):
  - Data transmitted: 2(N-1)/N × total_data
  - For N=4: 2(3)/4 = 1.5× total data

NVLink Bandwidth (H100):
  - Single direction: 50 GB/s
  - Bidirectional: 100 GB/s

Example for 100MB tensor:
  - Data transmitted: 150 MB
  - Expected time: 150 MB / 100 GB/s = 1.5 ms

Validation targets:
  - 7B model layer (output): ~50 MB
  - All-reduce time: ~0.75 ms (acceptable)
  - Overhead: <5% of computation time
```

---

## 5. Troubleshooting & Diagnostics

### 5.1 Common NCCL Issues

| Issue                | Symptoms                        | Diagnosis                   | Solution                         |
| -------------------- | ------------------------------- | --------------------------- | -------------------------------- |
| **Timeout**          | "Timeout waiting for operation" | Check network, slow GPU     | Increase timeout, check GPU load |
| **Rank Mismatch**    | Hang, rank stuck                | Wrong rank assignment       | Verify rank derivation           |
| **NVLink Disabled**  | Low bandwidth                   | Check NCCL_DEBUG output     | Enable NVLink, check hardware    |
| **Memory Leak**      | OOM after many operations       | Uncleaned buffers           | Add explicit cleanup             |
| **Version Conflict** | Initialization fails            | Mixed PyTorch/CUDA versions | Update packages consistently     |

### 5.2 Diagnostic Script

```python
def diagnose_nccl_setup():
    """
    Diagnose NCCL setup and report issues.
    """

    print("=== NCCL Diagnostics ===\n")

    # CUDA/GPU Info
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"NCCL Version: {torch.cuda.nccl.version()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print()

    # NVLink Info
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    print()

    # Test all-reduce
    if torch.cuda.device_count() >= 2:
        print("Testing all-reduce...")
        try:
            dist.init_process_group(backend='nccl', rank=0, world_size=1)

            # Single GPU test (should succeed)
            tensor = torch.ones(1000000, device='cuda:0')
            start = time.perf_counter()
            dist.all_reduce(tensor)
            elapsed = time.perf_counter() - start

            bandwidth = (1000000 * 4 / 1e9) / elapsed  # GB/s
            print(f"All-reduce (1M floats): {elapsed*1000:.2f}ms, "
                  f"Bandwidth: {bandwidth:.1f} GB/s")

            dist.destroy_process_group()
            print("✓ All-reduce test passed")

        except Exception as e:
            print(f"✗ All-reduce test failed: {e}")

    print("\n=== Environment Variables ===")
    for var in ['NCCL_DEBUG', 'NCCL_ALGO', 'NCCL_TIMEOUT',
                'NCCL_NVLINK_DISABLE', 'NCCL_SOCKET_IFNAME']:
        value = os.environ.get(var, 'not set')
        print(f"{var}: {value}")
```

---

## 6. NCCL Integration with Orchestrator

### 6.1 ProcessGroupManager NCCL Usage

```python
class ProcessGroupManager:
    """Excerpt showing NCCL integration."""

    def initialize_nccl(self, rank: int, world_size: int,
                       master_addr: str, master_port: int):
        """Initialize NCCL backend."""

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)

        # Export NCCL configuration
        os.environ['NCCL_DEBUG'] = 'WARN'
        os.environ['NCCL_ALGO'] = 'Ring'
        os.environ['NCCL_PROTO'] = 'LL'
        os.environ['NCCL_TIMEOUT'] = '600'

        try:
            dist.init_process_group(
                backend='nccl',
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=10)
            )

            # Log NCCL version
            logger.info(f"NCCL initialized: version {torch.cuda.nccl.version()}")

        except Exception as e:
            logger.error(f"NCCL initialization failed: {e}")
            raise

    def all_reduce(self, tensor: torch.Tensor):
        """Wrapper for all-reduce with error handling."""
        try:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        except Exception as e:
            logger.error(f"All-reduce failed: {e}")
            raise
```

---

## 7. Performance Targets

### 7.1 NCCL Latency Targets

| Operation  | Tensor Size | Target Latency | Notes                  |
| ---------- | ----------- | -------------- | ---------------------- |
| All-reduce | 100KB       | <0.5ms         | Small parameter sync   |
| All-reduce | 10MB        | ~1ms           | Gradient communication |
| All-reduce | 100MB       | ~10ms          | Large model layer      |
| Broadcast  | 100MB       | ~5ms           | Model distribution     |

### 7.2 Bandwidth Targets

```
Ring All-Reduce Bandwidth for 4 V100s:
Expected: 2(N-1)/N × NVLink_BW
        = 1.5 × 100 GB/s
        = 150 GB/s effective

Measured Target: >90 GB/s (90% efficiency)
```

---

## 8. Configuration Checklist

- [ ] NCCL 2.18+ installed and verified
- [ ] PyTorch compiled with NCCL support
- [ ] Network interface identified (NCCL_SOCKET_IFNAME)
- [ ] NVLink enabled and verified (NCCL_NVLINK_DISABLE=0)
- [ ] Timeout set appropriately (600 seconds)
- [ ] Logging configured (NCCL_DEBUG=WARN)
- [ ] Ring algorithm selected for throughput
- [ ] All-reduce latency benchmarked
- [ ] Fallback strategy documented (CPU fallback with Gloo)

---

## 9. Validation Criteria

- [ ] NCCL version verified (2.18+)
- [ ] All-reduce latency <1ms for 10MB tensors
- [ ] Bandwidth utilization >90% on ring
- [ ] 4-GPU scaling efficiency >95%
- [ ] Diagnostic script runs without errors
- [ ] NVLink properly detected and used
- [ ] Timeout handling works correctly
- [ ] Documentation complete and clear

---

## 10. Next Steps

1. **Task 1.1.1**: Finalize distributed architecture document
2. **Task 1.1.5**: Implement tensor parallelism with NCCL all-reduce
3. **Task 1.1.6**: Implement orchestrator using ProcessGroupManager
4. **Benchmarking**: Profile all-reduce latency on target hardware

---

## References

- NVIDIA NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- Baidu Ring All-Reduce Paper: https://arxiv.org/abs/1709.02929
- NVIDIA Collective Communications Library (NCCL) Whitepaper
- Horovod NCCL Integration
