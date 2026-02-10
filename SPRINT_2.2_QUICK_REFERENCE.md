---
title: "Sprint 2.2 Quick Reference"
date: "2025-12-26"
purpose: "Fast navigation and execution"
---

# Sprint 2.2 Quick Reference Guide

## 🎯 Mission (One-Liner)

**Build 1000+ req/sec distributed inference with paged attention + speculative decoding**

---

## 📂 Project Structure

```
PHASE2_DEVELOPMENT/src/
├── distributed/
│   └── engine.py              # Tensor/pipeline parallelism
├── cache/
│   └── manager.py             # Paged attention + prefix cache
├── speculative/
│   └── decoder.py             # Draft model + verification
├── batching/
│   └── token_batcher.py       # Token-level batching
├── serving/                   # (To be implemented)
│   ├── request_handler.py     # HTTP interface
│   └── unified_pipeline.py    # Component orchestration
└── perf/                      # (To be implemented)
    └── benchmarks.py          # Performance measurement
```

---

## 🏗️ Core Components

### 1. Distributed Inference Engine

**File**: `src/distributed/engine.py`

**Quick Start**:

```python
from distributed.engine import DistributedInferenceEngine, DistributedConfig

config = DistributedConfig(num_gpus=8, world_size=8)
engine = DistributedInferenceEngine(config)

# Shard model across GPUs
model = engine.shard_model(model)

# Run distributed inference
output = engine.distributed_forward(model, batch)
```

**Key Classes**:

- `DistributedInferenceEngine`: Main orchestrator
- `TensorShardManager`: Model sharding
- `CollectiveCommunicator`: GPU communication
- `GPUMemoryManager`: Memory management

---

### 2. KV Cache Manager

**File**: `src/cache/manager.py`

**Quick Start**:

```python
from cache.manager import PagedAttentionKVCache, PrefixCache, PageConfig

config = PageConfig(page_size=16)
kv_cache = PagedAttentionKVCache(config, num_pages=4096)

# Cache K,V for a sequence
kv_cache.write_kv(seq_id, k_tensor, v_tensor, token_pos)

# Retrieve from cache
k, v = kv_cache.read_kv(seq_id, start, end)

# Prefix caching
prefix_cache = PrefixCache(kv_cache)
prefix_cache.cache_prefix(tokens, k, v)
retrieved = prefix_cache.get_prefix(tokens)
```

**Key Classes**:

- `PagedAttentionKVCache`: Paged memory allocation
- `PrefixCache`: Prefix sharing system
- `GPUMemoryPool`: Memory pooling

---

### 3. Speculative Decoding

**File**: `src/speculative/decoder.py`

**Quick Start**:

```python
from speculative.decoder import (
    SpeculativeDecoder,
    DraftModel,
    SpeculationConfig
)

config = SpeculationConfig(
    max_speculation_depth=4,
    draft_model_ratio=0.4
)

decoder = SpeculativeDecoder(main_model, config)

# Generate with speculation
output = decoder.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7
)

print(f"Generated {output.num_tokens} tokens")
print(f"Acceptance rate: {output.acceptance_rate:.2%}")
print(f"Speedup: {output.num_iterations / output.num_verified:.2f}x")
```

**Key Classes**:

- `SpeculativeDecoder`: Main decoder
- `DraftModel`: Lightweight draft model
- `SpeculativeVerifier`: Token verification
- `AdaptiveSpeculation`: Depth adjustment

---

### 4. Token-Level Batcher

**File**: `src/batching/token_batcher.py`

**Quick Start**:

```python
from batching.token_batcher import TokenBatcher, TokenRequest

batcher = TokenBatcher(max_batch_size=128, max_batch_tokens=4096)

# Add requests
batcher.add_request(
    request_id="req_1",
    prompt_tokens=tokens,
    max_tokens=100,
    priority=5
)

# Get batches
while True:
    batch = batcher.get_batch()
    if batch is None:
        break

    # Process batch
    output = model(batch.tokens)

    # Mark requests complete
    for req_id in batch.request_ids:
        batcher.mark_completed(req_id)

# Stats
stats = batcher.get_stats()
```

**Key Classes**:

- `TokenBatcher`: Token-level batching
- `TokenRequest`: Request representation
- `TokenBatch`: Batch representation
- `RequestQueue`: Priority queue
- `BatchScheduler`: Scheduling strategies

---

## 🔄 Integration Flow

```
Request comes in
    ↓
TokenBatcher (prioritize + batch)
    ↓
DistributedInferenceEngine (shard across GPUs)
    ↓
KV Cache (retrieve cached values if prefix matched)
    ↓
SpeculativeDecoder (draft + verify for faster generation)
    ↓
Update KV Cache with new tokens
    ↓
Return response
```

---

## ✅ Implementation Checklist

### Foundation (✅ Complete - Day 1)

- [x] Distributed Engine (700 lines)
- [x] KV Cache Manager (650 lines)
- [x] Speculative Decoder (600 lines)
- [x] Token Batcher (500 lines)

### Phase 1 - Integration (🔄 Days 2-3)

- [ ] Create module **init**.py files
- [ ] Write comprehensive test suite (100+ tests)
- [ ] Build request handler
- [ ] Implement unified pipeline
- [ ] Run initial benchmarks

### Phase 2 - Optimization (🔄 Days 4-8)

- [ ] Advanced KV cache features
- [ ] Speculative decoding tuning
- [ ] Batching optimization
- [ ] Load balancing

### Phase 3 - Production (🔄 Day 9)

- [ ] Full integration testing
- [ ] Performance validation
- [ ] Documentation
- [ ] Deployment preparation

---

## 📊 Performance Targets

| Metric       | Target        | How to Achieve                                 |
| ------------ | ------------- | ---------------------------------------------- |
| Throughput   | 1000+ req/sec | Token batching + speculative decoding          |
| P99 Latency  | <100ms        | Continuous batching + priority scheduling      |
| Memory/Token | <500MB        | Paged attention + prefix caching               |
| GPU Util     | >85%          | Token-level batching + speculative parallelism |

---

## 🧪 Testing Strategy

### Unit Tests (per component)

```python
# tests/test_distributed.py
- test_tensor_sharding()
- test_memory_management()
- test_collective_operations()

# tests/test_cache.py
- test_paged_allocation()
- test_prefix_caching()
- test_memory_pooling()

# tests/test_speculative.py
- test_draft_generation()
- test_verification()
- test_acceptance_sampling()

# tests/test_batching.py
- test_token_batching()
- test_priority_queue()
- test_scheduling()
```

### Integration Tests

```python
# End-to-end pipeline
- test_e2e_inference()
- test_memory_efficiency()
- test_throughput()
- test_latency()
```

---

## 🔍 Debugging Tips

### Distributed Engine Issues

```python
# Check sharding
engine.shard_manager.shard_linear_weight(weight)

# Check communication
communicator.all_reduce_sum(tensor)

# Memory stats
stats = engine.memory_manager.get_memory_stats()
```

### KV Cache Issues

```python
# Check allocation
kv_cache.allocate_pages(4, "seq_1")

# Check retrieval
k, v = kv_cache.read_kv("seq_1", 0, 10)

# Memory stats
stats = kv_cache.get_memory_stats()
```

### Speculative Decoding Issues

```python
# Check draft generation
draft_ids = draft_model.generate_draft(input_ids, 4)

# Check verification
verified, rate = verifier.verify_tokens(...)

# Check acceptance rate
if acceptance_rate < 0.5:
    # Lower speculation depth
    config.max_speculation_depth -= 1
```

### Batching Issues

```python
# Check queue size
pending = batcher.get_pending_count()

# Check batch size
batch = batcher.get_batch()
print(f"Batch: {batch.batch_size} requests, {batch.total_tokens} tokens")

# Check stats
stats = batcher.get_stats()
```

---

## 📚 Key Formulas

### Paged Attention Memory

```
Total Memory = num_pages × page_size × elements_per_page × bytes_per_element
             = 4096 × 16 × 8192 × 2 (float16)
             = ~4.3 GB
```

### Speculative Speedup

```
Speedup ≈ (1 + speculation_depth × acceptance_rate)
        ≈ (1 + 4 × 0.75) = 4x (with 75% acceptance)
```

### Token Batching Throughput

```
Throughput = (batch_size × tokens_per_seq) / latency_per_batch
           = (128 × 512) / 0.1s ≈ 655k tokens/sec
```

---

## 🚀 Quick Commands

### Run Tests

```bash
cd PHASE2_DEVELOPMENT
pytest tests/test_distributed.py -v
pytest tests/test_cache.py -v
pytest tests/ -v --cov=src
```

### Profile Performance

```python
from src.perf.benchmarks import benchmark_pipeline

results = benchmark_pipeline(model, num_requests=100)
print(f"Throughput: {results['throughput']} req/sec")
print(f"P99 Latency: {results['p99_latency_ms']} ms")
```

### Monitor GPU

```bash
nvidia-smi watch -n 0.5
```

---

## 📞 Quick Help

**What file do I edit for...**

| Task           | File                                      |
| -------------- | ----------------------------------------- |
| GPU sharding   | `src/distributed/engine.py`               |
| KV cache       | `src/cache/manager.py`                    |
| Draft model    | `src/speculative/decoder.py`              |
| Token batching | `src/batching/token_batcher.py`           |
| HTTP interface | `src/serving/request_handler.py` (future) |
| Tests          | `tests/test_*.py`                         |
| Benchmarks     | `src/perf/benchmarks.py` (future)         |

---

## 🎯 Daily Goals

### Day 2 (Tomorrow)

- [ ] Add **init**.py for all modules
- [ ] Write 20 distributed engine tests
- [ ] Write 20 cache tests
- [ ] Basic request handler sketch

### Day 3

- [ ] 20 speculative decoding tests
- [ ] 20 batching tests
- [ ] Unified pipeline implementation
- [ ] Initial E2E test

### Days 4-9

- Follow implementation plan in SPRINT_2.2_KICKOFF.md

---

## 💡 Pro Tips

1. **Always profile before optimizing** - Use nvidia-smi and Python profilers
2. **Test in isolation** - Test components separately before integration
3. **Measure impact** - Benchmark after every significant change
4. **Log everything** - Use detailed logging for debugging distributed issues
5. **Read papers** - vLLM and Megatron-LM papers have great insights

---

## 📖 Documentation Links

- **Architecture**: See SPRINT_2.2_KICKOFF.md
- **Status**: See SPRINT_2.2_STATUS_DAY1.md
- **Code Docs**: Inline docstrings in each module
- **Papers**: References in SPRINT_2.2_KICKOFF.md

---

## 🎓 Learning Resources

**For distributed training**:

- Megatron-LM paper & code
- PyTorch distributed documentation

**For inference optimization**:

- vLLM paper & code
- TensorRT-LLM repository

**For speculative decoding**:

- DeepMind paper
- Implementation in vLLM

---

**Last Updated**: 2025-12-26  
**Sprint Phase**: Foundation ✅ → Integration 🔄  
**Next Review**: Day 2 End

🚀 **Ready to build!**
