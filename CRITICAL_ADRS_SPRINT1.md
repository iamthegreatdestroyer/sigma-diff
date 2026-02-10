# CRITICAL ADRS FOR PHASE 3 SPRINT 1

**Status**: REQUIRED COMPLETION BY DEC 27, 2025  
**Purpose**: Unblock Sprint 1.2 & 1.3 implementation  
**Owner**: @ARCHITECT with @APEX input

---

## ADR-002: KV-Cache Compression & Distribution Strategy

**Status**: REQUIRED (blocks Sprint 1.2)  
**Priority**: P0  
**Decision Date**: Must decide by Dec 27, 2025

### Context

Sprint 1.2 targets 40-50% KV-cache memory reduction via fp8 compression, but the strategy is underspecified:

1. **When is compression applied?**

   - Immediately after cache write? (latency cost)
   - Lazily when cache fills? (complexity)
   - Only for historical tokens (keep recent in fp32)? (hybrid)

2. **Decompression overhead?**

   - Must decompress before attention computation
   - Latency budget: <1ms per decompression
   - Need to validate achievable on A100/H100

3. **Interaction with distributed sharding?**

   - Each GPU holds 1/4 of cache (head-wise)
   - Can compress independently (no coordination needed)
   - Validation: Compression doesn't break distributed semantics

4. **Quality degradation?**
   - fp8 typically causes <0.5% perplexity loss
   - Need to benchmark actual Llama2 model

### Decision Options

#### Option A: Immediate Compression (Simple, High Latency Cost)

```
When token writes to KV-cache:
  1. Store in fp32 (warm cache)
  2. Immediately compress to fp8
  3. On attention: decompress fp8 → fp32 → compute

Pros:
  • Simplest implementation (no state tracking)
  • Predictable memory usage

Cons:
  • Compression/decompression on every attention op
  • Estimated latency cost: ~2-3ms per token
  • P99 latency impact: -5-10ms
  • Speedup target (3.8x) may not be achievable
```

#### Option B: Lazy Compression (Complex, Low Latency Cost)

```
Cache structure: [recent_fp32] | [old_fp8]

When token writes:
  1. Store in fp32 (hot zone)
  2. When hot zone fills (16 tokens):
     - Compress hot tokens to fp8
     - Move to cold zone
  3. On attention:
     - Recent tokens: use fp32 directly
     - Historical tokens: decompress from fp8

Pros:
  • Minimal latency cost (few recent tokens stay fp32)
  • Compression amortized across many tokens

Cons:
  • Complex state tracking (two zones)
  • Edge case: what if sequence is shorter than hot zone?
  • Validation complexity increases
```

#### Option C: Hybrid per-layer Compression (Balanced)

```
Strategy: Compress only in late transformer layers

Rationale:
  • Early layers: low-attention importance to historical tokens
  • Late layers: high-importance to full context
  • Compress early layer cache, keep late layer fp32

Memory savings: ~30% (vs 50% if all layers compressed)
Latency impact: Minimal (late layers stay fp32)

Pros:
  • Simple to implement
  • Minimal latency cost
  • Reasonable memory savings

Cons:
  • Less memory savings than Option A
  • Per-layer decisions add complexity
```

### Recommendation: **Option B (Lazy Compression)** - If Team Can Execute

**Rationale**:

- Meets 40% memory reduction target
- Minimal latency impact (<1ms)
- Scales naturally to longer sequences

**Risk**: Implementation complexity 2-3× higher than Option A

### Fallback: **Option C (Hybrid)** - If Confidence Low

**Rationale**:

- Simpler than Option B
- Still achieves 30-40% memory savings
- Minimal latency impact
- Can upgrade to Option B later

### Decision Template

```
Chosen Option: [A/B/C]

Justification:
[Why this option chosen over others]

Decompression Latency Budget:
[Allocated latency in inference path]

Validation Plan:
[How will we verify compression doesn't break inference?]

Benchmark Plan:
[Perplexity loss measurement on Llama2-7B]

Quality Acceptance Criteria:
- Perplexity loss: <0.5%
- Latency overhead: <1ms per attention op
- Memory reduction: >40%

Implementation Timeline:
[Estimate LOC, complexity, dependencies]
```

---

## ADR-003: Load Balancing & Request Routing Strategy

**Status**: REQUIRED (blocks Sprint 1.3)  
**Priority**: P0  
**Decision Date**: Must decide by Dec 27, 2025

### Context

Load balancer must coordinate with distributed KV-cache, but strategy is vague.

**The Core Tension**:

- **Stateless Ideal**: Route any request to any GPU (pure LB)
- **Stateful Reality**: KV-cache tied to specific GPU rank (context dependent)
- **Use Case**: Multi-turn conversations need context preservation

### Decision Options

#### Option A: Sticky Sessions (Simple, Slightly Imbalanced)

```
Algorithm:
  hash_key = user_id or session_id
  preferred_gpu = hash(hash_key) % num_gpus

  1. Route request to preferred_gpu
  2. If preferred_gpu overloaded:
     - Queue on preferred_gpu (wait for capacity)
     - Never route to different GPU
  3. Response includes session_id for next request

Semantics:
  • Same user → same GPU (KV-cache preserved)
  • Different user → may route to different GPU
  • Warm cache maintained across multi-turn conversations

Pros:
  • Implementation: ~50 LOC (simple hash + routing table)
  • No cache distribution needed
  • Context always available (correctness guaranteed)

Cons:
  • Load imbalance: user-dependent (could be 10-20%)
  • Unfair: heavy users block light users on same GPU
  • Fairness issue: GPU 0 handles 30% requests, GPU 3 handles 10%
```

#### Option B: Distributed Context Cache (No Imbalance, High Complexity)

```
Strategy: Maintain copy of KV-cache on all GPUs

When request comes to GPU 0:
  1. GPU 0 computes attention (reads local cache)
  2. Broadcast KV-cache updates to GPU 1, 2, 3
  3. All GPUs maintain identical cache state

Next request from same user to GPU 1:
  1. GPU 1 has identical cache (warm start)
  2. Context preserved (correctness guaranteed)

Pros:
  • Pure load balancing (any GPU can handle any request)
  • No fairness issues (perfect distribution possible)

Cons:
  • Memory overhead: 3× cache size (store on all GPUs)
  • Communication: Broadcast cache after each request
  • Latency: Additional ~10-15ms sync overhead
  • Complexity: Cache coherency protocol needed
  • Cost: 3× GPU memory budget spent on cache
```

#### Option C: State Server (Centralized Cache, Medium Complexity)

```
Strategy: Separate service manages KV-cache

Architecture:
  ┌─────────────────────────────┐
  │   Request Router (stateless) │
  └──────────────┬──────────────┘
                 │ routes to best GPU
  ┌──────────────▼──────────────┐
  │   GPU 0, 1, 2, 3 (inference)│
  └──────────┬──────────────────┘
             │ request cache
  ┌──────────▼──────────────┐
  │  KV-Cache Server (Redis)│
  │  (shared, external)     │
  └─────────────────────────┘

When GPU 0 processes request:
  1. Fetch cache from Redis
  2. Compute attention
  3. Update cache in Redis

When GPU 3 processes same request:
  1. Fetch updated cache from Redis
  2. Context is preserved (correct)

Pros:
  • Pure load balancing (any GPU → any request)
  • Solves imbalance problem

Cons:
  • New operational component (Redis, reliability risk)
  • Network latency: Redis fetch ~5-10ms per request
  • Cache coherency bugs possible
  • Adds operational complexity
```

#### Option D: Hybrid Session-Aware LB (Balanced)

```
Algorithm:
  preferred_gpu = hash(user_id) % num_gpus

  Route to preferred_gpu if:
    • Queue length < threshold (2 requests)

  Else (preferred GPU busy):
    • Route to least-loaded GPU
    • Expect KV-cache miss on cold GPU
    • Cold GPU re-fetches context from state server (Option C hybrid)

Pros:
  • Usually sticky (use cached context)
  • Gracefully degrades when GPU overloaded
  • Reasonable balance

Cons:
  • Complexity: hybrid sticky + state server
  • Edge case: thrashing if all GPUs overloaded
```

### Decision Criteria Matrix

| Criterion               | Option A | Option B | Option C | Option D |
| ----------------------- | -------- | -------- | -------- | -------- |
| Implementation LOC      | 50       | 500      | 300      | 400      |
| Memory overhead         | 1×       | 4×       | Minimal  | 1.5×     |
| Load imbalance          | 10-20%   | 0-5%     | 0-5%     | 5-10%    |
| Latency penalty         | 0ms      | +15ms    | +5ms     | +2ms     |
| Operational complexity  | Low      | High     | High     | Medium   |
| Correctness guarantee   | Strong   | Strong   | Medium   | Medium   |
| Recommended for Phase 3 | ✅       | ❌       | ⚠️       | ⚠️       |

### Recommendation: **Option A (Sticky Sessions)**

**Rationale for Phase 3**:

1. Multi-turn conversation is primary use case (not random requests)
2. Load imbalance acceptable (users have natural distribution)
3. Simplicity critical for team velocity
4. Can upgrade to Option C (state server) in Phase 3.2 if needed

**Acceptance Criteria**:

- Load imbalance <10% across GPUs (monitor and validate)
- Context preserved across multi-turn conversations
- Session routing deterministic (same user_id → same GPU)
- Failover when preferred GPU unavailable (<100ms recovery)

### Fallback: **Option D (Hybrid)** - If Option A Shows Imbalance

**When to Upgrade**: If monitoring shows >15% load imbalance or users complain about slowness on peak GPU.

### Decision Template

```
Chosen Option: [A/B/C/D]

Session Identity:
[How is user_id determined? Request header? Auth token? Cookie?]

Load Balancing Algorithm:
[Pseudocode for request routing]

Failover Strategy:
[When preferred GPU unavailable, what happens?]

Imbalance Acceptance Threshold:
[Acceptable load imbalance: <10%? <15%?]

Cache Correctness Guarantee:
[When is context preserved? When is it lost?]

Monitoring & Alerts:
[How do we detect imbalance? Alert thresholds?]

Upgrade Path:
[If Option A shows problems, how do we migrate to Option C?]
```

---

## ADR-004: Distributed Debugging & Observability (Important, Not Blocking)

**Status**: REQUIRED BEFORE Sprint 2  
**Priority**: P1  
**Decision Date**: Must decide by Jan 10, 2026 (start of Sprint 2)

### Context

Distributed inference introduces debugging challenges not present in single-GPU systems:

1. **Rank-Specific Behavior**: Different GPUs may behave differently (NVLink topology, memory pressure, thermal throttling)
2. **Timing-Dependent Bugs**: NCCL race conditions, communication hangs, deadlocks
3. **Multi-Request Correlation**: How do we trace a request through 4 GPUs?
4. **Performance Profiling**: Which GPU is bottleneck? Where is time spent?

### Key Decisions Needed

1. **Log Aggregation**:

   - Centralized: All ranks → single file
   - Distributed: Each rank has local log
   - Hybrid: Local buffer → periodic flush to central location

2. **Rank Identification**:

   - How do logs include rank information?
   - How do we query logs by rank?
   - How do we correlate ranks across request?

3. **Tracing & Causality**:

   - Can we trace request → GPU 0 → GPU 1 → GPU 2?
   - How do we detect communication bottlenecks?
   - How do we measure per-rank latencies?

4. **Profiling Tools**:
   - Which profilers work with distributed PyTorch?
   - How do we collect NCCL communication profiles?
   - How do we visualize communication timeline across ranks?

### Recommendation Template

```
Log Aggregation Strategy:
[Centralized/Distributed/Hybrid with rationale]

Structured Logging Format:
[Example log entry with rank, timestamp, duration fields]

Tracing Framework:
[OpenTelemetry? Custom? Which tool?]

Profiling Tools:
[PyTorch profiler? NCCL profiler? Custom?]

Developer Workflow for Debugging:
[Step-by-step: How to debug a hung request?]

Example: Root Cause Analysis
[Walk through finding bottleneck in distributed inference]
```

---

## ADR-005: Failure Modes & Recovery Strategy

**Status**: REQUIRED BEFORE Sprint 3  
**Priority**: P1  
**Decision Date**: Start of Sprint 2, finalize by start of Sprint 3

### Context

Distributed system introduces failure modes not present in single-GPU inference:

1. GPU OOM mid-inference
2. Rank hangs (NCCL deadlock)
3. NVLink hardware failure
4. Process crash (segfault, exception)
5. Network connectivity loss (multi-node future)

### Key Decisions Needed

1. **Detection**: How quickly can we detect each failure?
2. **Recovery**: Can we recover, or must restart?
3. **User Impact**: Do requests fail immediately or queue?
4. **SLA Impact**: What's the recovery time budget?

### Recommendation Template

```
Failure Mode Taxonomy:
- GPU OOM: Detection (GPU memory error), Recovery (reduce batch, restart)
- Rank Hang: Detection (timeout 30s), Recovery (graceful shutdown)
- Process Crash: Detection (exit code), Recovery (supervisor restart)
[... more modes]

Detection & Recovery Matrix:
[Per failure mode: detection method, recovery action, SLA impact]

Checkpoint Recovery Procedure:
[If we restart, how do we resume? Saved checkpoint format?]

Chaos Engineering Tests:
[Which failure modes will we test before production?]
```

---

## Summary: ADR Completion Timeline

| ADR     | Title                                 | Due Date | Owner      | Impact      |
| ------- | ------------------------------------- | -------- | ---------- | ----------- |
| ADR-002 | KV-Cache Compression Strategy         | Dec 27   | @APEX      | Blocks S1.2 |
| ADR-003 | Load Balancing & Request Routing      | Dec 27   | @ARCHITECT | Blocks S1.3 |
| ADR-004 | Distributed Debugging & Observability | Jan 10   | @SENTRY    | Critical    |
| ADR-005 | Failure Modes & Recovery              | Jan 20   | @FORTRESS  | SLA-related |

---

**Next Step**: Schedule ADR review meetings (Dec 23-27) with team members who will implement decisions.
