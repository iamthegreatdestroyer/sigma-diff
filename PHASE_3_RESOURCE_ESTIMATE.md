# PHASE 3 RESOURCE ESTIMATE & TIMELINE

## Realistic Planning for 6-Month Delivery

**Document Version:** 1.0  
**Date:** December 20, 2025  
**Planning Horizon:** Q2 2026 (January - June)

---

## EXECUTIVE SUMMARY

**Phase 3 Resource Plan:**

- **Duration:** 24 weeks (6 months)
- **Team Size:** 6 core engineers + support
- **Total Cost:** $250K-$350K
- **Dependencies:** Phase 2 complete (✅)
- **Critical Path:** Distributed engine → Batching → Quantization framework
- **Key Risk:** Distributed sync overhead (mitigated by careful design)

### Timeline Overview

```
Q1 2026 (Jan-Mar)              Q2 2026 (Apr-Jun)
├─ Week 1-4: Foundations       ├─ Week 9-12: Ecosystem
├─ Week 5-8: Core Features     ├─ Week 13-16: Production
├─ Final: Integration Tests    └─ Final: Release v3.0
```

---

## PART 1: TEAM STRUCTURE & ALLOCATION

### Core Team (6 FTE)

```
┌──────────────────────────────────────────────────────────┐
│                   PHASE 3 TEAM STRUCTURE                 │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  Engineering Manager (0.5 FTE)                           │
│  └─ Coordinates sprints, unblocks issues, reports       │
│                                                            │
│  Backend Team Lead / @APEX (1.0 FTE)                    │
│  ├─ Distributed executor (primary)                      │
│  ├─ Multi-node coordination                             │
│  ├─ gRPC integration                                     │
│  └─ Failover + recovery mechanisms                      │
│                                                            │
│  Performance Engineer / @VELOCITY (1.0 FTE)             │
│  ├─ Quantization framework + strategies                 │
│  ├─ Continuous batching engine                          │
│  ├─ KV cache compression                                │
│  ├─ Benchmarking + profiling                            │
│  └─ Performance optimization (ongoing)                  │
│                                                            │
│  Architect / @ARCHITECT (0.8 FTE)                       │
│  ├─ Sparse attention implementation                     │
│  ├─ Architecture decisions                              │
│  ├─ Design reviews                                       │
│  ├─ Multi-model orchestration                           │
│  └─ Documentation + diagrams                            │
│                                                            │
│  ML Engineer / @TENSOR (0.8 FTE)                        │
│  ├─ QLoRA fine-tuning system                            │
│  ├─ Training loop + optimization                        │
│  ├─ HuggingFace model loader                            │
│  ├─ Accuracy validation                                 │
│  └─ Model conversion tools                              │
│                                                            │
│  API/Integration Engineer / @SYNAPSE (0.8 FTE)          │
│  ├─ Request router + load balancing                     │
│  ├─ OpenAI API compatibility                            │
│  ├─ Format converters                                   │
│  ├─ Integration testing                                 │
│  └─ API documentation                                   │
│                                                            │
│  QA/Test Lead / @ECLIPSE (0.8 FTE)                      │
│  ├─ Test strategy + planning                            │
│  ├─ Integration testing                                 │
│  ├─ Continuous benchmarking                             │
│  ├─ Error handling verification                         │
│  ├─ Production hardening                                │
│  └─ Release validation                                  │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

### Support Team (Part-Time)

| Role                 | Allocation | Responsibility                         |
| -------------------- | ---------- | -------------------------------------- |
| **DevOps**           | 0.3 FTE    | CI/CD setup, Docker, deployment        |
| **Technical Writer** | 0.3 FTE    | Documentation, API docs, guides        |
| **Product Manager**  | 0.2 FTE    | Prioritization, stakeholder management |
| **Security**         | 0.1 FTE    | Code review, security assessment       |

**Total: 6.2 FTE core + 0.9 FTE support = 7.1 FTE**

---

## PART 2: DETAILED SPRINT BREAKDOWN

### SPRINT 1: Foundation (Weeks 1-2)

**Theme:** Establish distributed architecture & continuous batching base

**Goals:**

- ✅ Distributed executor skeleton
- ✅ Request router
- ✅ Quantization framework interface
- ✅ Continuous batching scheduler

**Team Assignments:**

| Engineer  | Component                 | Hours | Status                  |
| --------- | ------------------------- | ----- | ----------------------- |
| @APEX     | Distributed executor v0.1 | 80h   | Design → Impl → Testing |
| @VELOCITY | Batching scheduler v0.1   | 80h   | Design → Impl → Testing |
| @SYNAPSE  | Request router            | 60h   | Design → Impl → Testing |
| @VELOCITY | Quant framework           | 40h   | Interface → Integration |
| @ECLIPSE  | Test infrastructure       | 40h   | Setup → Mocking         |

**Deliverables:**

- `src/core/distributed/executor.cpp` (skeleton)
- `src/api/router.cpp` (basic version)
- `src/core/engine/scheduler.cpp` (scheduler interface)
- `src/core/quantization/framework.h` (plugin interface)
- 20+ unit tests

**Definition of Done:**

- ✅ Components integrate
- ✅ No compiler warnings
- ✅ Basic tests pass
- ✅ Performance benchmarked

---

### SPRINT 2: Multi-Node Integration (Weeks 3-4)

**Theme:** Complete distributed coordination, enable 2+ node inference

**Goals:**

- ✅ Node manager + health checks
- ✅ KV cache distribution protocol
- ✅ Continuous batching v1.0 complete
- ✅ Multi-node load balancing

**Team Assignments:**

| Engineer   | Component                  | Hours | Status                   |
| ---------- | -------------------------- | ----- | ------------------------ |
| @APEX      | Distributed executor v1.0  | 80h   | Testing → Optimization   |
| @VELOCITY  | Continuous batching v1.0   | 80h   | Completion → Performance |
| @ARCHITECT | Design review + guidance   | 40h   | Code review → ADRs       |
| @ECLIPSE   | Multi-node testing         | 60h   | Test suite → Validation  |
| @SYNAPSE   | Load balancer optimization | 40h   | Performance tuning       |

**Deliverables:**

- Complete distributed executor
- Continuous batching engine (production-ready)
- Node manager + failover
- 40+ integration tests
- Performance benchmarks (2-4 node scaling)

**Acceptance Criteria:**

```
Performance:
├─ 2-node: 1.8× throughput
├─ 4-node: 3.2× throughput
├─ Failover: <2 sec recovery
└─ Latency: <200ms P99

Code Quality:
├─ 0 compiler warnings
├─ Test coverage >85%
├─ Backward compatible
└─ Documentation complete
```

**Sprint Velocity:** ~280 engineer-hours → Tier 1 core complete

---

### SPRINT 3: Quantization Ecosystem (Weeks 5-6)

**Theme:** Implement diverse quantization strategies

**Goals:**

- ✅ GPTQ strategy complete
- ✅ AWQ strategy complete
- ✅ Auto-selector intelligent
- ✅ Calibration pipeline

**Team Assignments:**

| Engineer  | Component            | Hours | Status                        |
| --------- | -------------------- | ----- | ----------------------------- |
| @VELOCITY | GPTQ implementation  | 100h  | Algorithm → Impl → Opt        |
| @VELOCITY | AWQ implementation   | 100h  | Algorithm → Impl → Opt        |
| @VELOCITY | Auto-selector        | 40h   | Logic → Training → Validation |
| @TENSOR   | Calibration pipeline | 40h   | Data handling → Accuracy      |
| @ECLIPSE  | Quantization testing | 80h   | Accuracy → Performance tests  |

**Deliverables:**

- GPTQ strategy (800+ lines C++)
- AWQ strategy (800+ lines C++)
- Auto-selector (300+ lines)
- 50+ quantization tests
- Benchmark suite (5+ models)

**Acceptance Criteria:**

```
Accuracy:
├─ GPTQ: <0.5% loss on MMLU
├─ AWQ: <0.3% loss on MMLU
├─ Auto-selector: ≥80% correct
└─ Fallbacks: All paths work

Performance:
├─ Quantization: <1 min per 7B model
├─ Inference: 95% vs fastest strategy
└─ Memory: 25% of FP32 baseline
```

**Sprint Velocity:** ~260 engineer-hours → Major quantization expansion

---

### SPRINT 4: Long Context & Performance (Weeks 7-8)

**Theme:** Enable extended context windows

**Goals:**

- ✅ Sparse attention patterns complete
- ✅ KV cache compression working
- ✅ 32K token context enabled
- ✅ Streaming attention

**Team Assignments:**

| Engineer   | Component                        | Hours | Status                        |
| ---------- | -------------------------------- | ----- | ----------------------------- |
| @ARCHITECT | Sparse attention (local)         | 80h   | Implementation → Testing      |
| @ARCHITECT | Sparse attention (strided/block) | 80h   | Implementation → Testing      |
| @VELOCITY  | KV cache compression             | 80h   | Implementation → Optimization |
| @TENSOR    | Streaming attention              | 40h   | Integration → Validation      |
| @ECLIPSE   | Long-context testing             | 100h  | Benchmark suite → Validation  |

**Deliverables:**

- Sparse attention implementation (1000+ lines)
- KV cache compression (600+ lines)
- Streaming attention handler (300+ lines)
- 60+ long-context tests
- Benchmark suite (context scaling)

**Acceptance Criteria:**

```
Context Scaling:
├─ 4K baseline: 100% accuracy
├─ 8K: 80% accuracy, 2× context
├─ 16K: 90% accuracy, 4× context
├─ 32K: 85% accuracy, 8× context

Performance:
├─ Memory: <100MB for 32K
├─ Latency: O(n·sqrt(n)) or better
└─ Throughput: No regression
```

**Sprint Velocity:** ~300 engineer-hours → Major capability expansion

---

### SPRINT 5: Fine-Tuning & Model Support (Weeks 9-10)

**Theme:** Enable customization & ecosystem integration

**Goals:**

- ✅ QLoRA framework complete
- ✅ HuggingFace loader working
- ✅ Format converters (GGUF, SafeTensors)
- ✅ 20+ supported models

**Team Assignments:**

| Engineer | Component              | Hours | Status                        |
| -------- | ---------------------- | ----- | ----------------------------- |
| @TENSOR  | QLoRA adapter          | 100h  | Design → Impl → Testing       |
| @TENSOR  | Training loop          | 80h   | Implementation → Optimization |
| @TENSOR  | HF loader              | 60h   | Integration → Validation      |
| @SYNAPSE | Format converters      | 60h   | Implementation → Testing      |
| @ECLIPSE | Fine-tuning validation | 100h  | Accuracy → Speed tests        |

**Deliverables:**

- QLoRA framework (700+ lines Python)
- Training loop + optimizer (500+ lines)
- HuggingFace loader (500+ lines)
- Format converters (700+ lines)
- 60+ model loading tests
- Training benchmark suite

**Acceptance Criteria:**

```
Fine-Tuning:
├─ Memory: <4GB peak (7B)
├─ Speed: <1 hour (7B)
├─ Quality: 95% of full fine-tuning
└─ LoRA size: <50MB

Model Support:
├─ HF models: ≥20 architectures
├─ Format support: GGUF, SafeTensors, PyTorch
├─ Load time: <5 min (7B)
└─ Auto-quantization: ≥80% accuracy
```

**Sprint Velocity:** ~400 engineer-hours → Major ecosystem expansion

---

### SPRINT 6: Multi-Model & Hardening (Weeks 11-12)

**Theme:** Production readiness

**Goals:**

- ✅ Multi-model orchestration
- ✅ Production hardening
- ✅ Error handling complete
- ✅ Monitoring + logging

**Team Assignments:**

| Engineer   | Component                | Hours | Status                       |
| ---------- | ------------------------ | ----- | ---------------------------- |
| @ARCHITECT | Multi-model orchestrator | 80h   | Design → Impl → Testing      |
| @ECLIPSE   | Error handling           | 80h   | Comprehensive review → Fix   |
| @ECLIPSE   | Monitoring + logging     | 60h   | Instrumentation → Validation |
| @SYNAPSE   | API polish               | 40h   | Edge cases → Refinement      |
| All        | Integration testing      | 80h   | End-to-end scenarios         |

**Deliverables:**

- Multi-model orchestrator (600+ lines)
- Error handling framework (400+ lines)
- Monitoring + logging (500+ lines)
- 80+ integration tests
- Release documentation
- Performance baselines (all features)

**Acceptance Criteria:**

```
Production Ready:
├─ 0 compiler warnings
├─ Test coverage >90%
├─ Error paths: 100% handled
├─ MTBF: >1000 hours
├─ Graceful degradation: Working
└─ Documentation: ≥95% complete

Reliability:
├─ Multi-model: 2-3 concurrent
├─ Memory: <15% overhead
├─ Switching: <500ms latency
└─ No crashes on overload
```

**Sprint Velocity:** ~340 engineer-hours → Production polish

---

### SPRINT 7: Testing & Release (Weeks 13-14)

**Theme:** Final validation, release preparation

**Goals:**

- ✅ Comprehensive testing
- ✅ Performance benchmarking
- ✅ Documentation complete
- ✅ Release candidate v3.0

**Team Assignments:**

| Engineer   | Component                | Hours | Status                             |
| ---------- | ------------------------ | ----- | ---------------------------------- |
| @ECLIPSE   | Release test suite       | 120h  | Comprehensive testing → Validation |
| @APEX      | Distributed stress tests | 60h   | Multi-day inference → Stability    |
| All        | Performance benchmarking | 100h  | Comprehensive metrics → Reporting  |
| @ARCHITECT | Documentation review     | 40h   | Completeness → Clarity             |
| @ECLIPSE   | Release validation       | 80h   | Checklist → Sign-off               |

**Deliverables:**

- 100+ integration tests
- Performance benchmark report
- Complete API documentation
- Architecture + design guides
- Troubleshooting guide
- Migration guide (Phase 2 → Phase 3)
- Release notes + changelog

**Definition of Done:**

- ✅ All Tier 1 + Tier 2 features complete
- ✅ Test pass rate >98%
- ✅ Performance targets met
- ✅ Documentation complete
- ✅ Release sign-off obtained

---

## PART 3: TIMELINE & CRITICAL PATH

### Gantt Chart (Simplified)

```
SPRINT 1  (Weeks 1-2)   ████░░░░░░░░░░░░ Foundation
SPRINT 2  (Weeks 3-4)   ████████░░░░░░░░ Distribution
SPRINT 3  (Weeks 5-6)   ████████████░░░░ Quantization
SPRINT 4  (Weeks 7-8)   ████████████████ Long Context
         (Weeks 9)     ████████████████ Fine-tuning (starts)
SPRINT 5  (Weeks 9-10)  ████████████████ Model Support
SPRINT 6  (Weeks 11-12) ████████████████ Hardening
SPRINT 7  (Weeks 13-14) ████████░░░░░░░░ Release Prep

Total: 14 weeks (9.5 months for features + 4.5 weeks final testing/release)
```

### Critical Path

```
START
  ↓
Feature 1.1 (Distributed Executor) ────────────┐
Feature 1.2 (Request Router)                    │
Feature 1.4 (Quant Framework) ─────────────────┼─→ Feature 2.1 (GPTQ)
Feature 1.3 (Continuous Batching) ─────────────┤   Feature 2.3 (Sparse)
                                                 │   Feature 3.1 (HF Loader)
                                                 │
                                                 ↓
                                           Feature 3.4 (Hardening)
                                                 ↓
                                           Feature 3.3 (Orchestration)
                                                 ↓
                                           RELEASE v3.0
```

**Critical path duration:** 12-14 weeks (features) + 2 weeks (release) = 14-16 weeks
**Total Phase 3:** 16-20 weeks = 4-5 months minimum for feature-complete release

### Dependency Map

```
┌─────────────────────┐
│ Phase 2 Complete ✓  │
└──────────┬──────────┘
           ↓
    ┌──────────────┐
    │ Tier 1: CORE │  (Weeks 1-4)
    │ - Executor   │
    │ - Router     │
    │ - Batching   │
    │ - Framework  │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Tier 2: FEAT │  (Weeks 5-8)
    │ - GPTQ/AWQ   │
    │ - Sparse     │
    │ - KV Cache   │
    │ - LoRA       │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Tier 3: ECO  │  (Weeks 9-12)
    │ - HF Loader  │
    │ - Converters │
    │ - Orches.    │
    │ - Hardening  │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Release v3.0 │  (Weeks 13-16)
    │ - Testing    │
    │ - Docs       │
    │ - Sign-off   │
    └──────────────┘
```

---

## PART 4: RISK MITIGATION & BUFFERS

### Schedule Buffers

| Phase               | Duration     | Buffer             | Rationale            |
| ------------------- | ------------ | ------------------ | -------------------- |
| Feature Development | 12 weeks     | +2 weeks (17%)     | Complexity unknowns  |
| Integration Testing | 2 weeks      | +1 week (50%)      | Cross-feature issues |
| Release Prep        | 2 weeks      | +1 week (50%)      | Documentation gaps   |
| **Total**           | **16 weeks** | **+4 weeks (25%)** | Safe delivery        |

**Recommended Timeline:** 20 weeks = 5 months (safe) or 16 weeks = 4 months (aggressive)

### Risk-Mitigation Activities

| Risk                           | Mitigation                          | Timeline   |
| ------------------------------ | ----------------------------------- | ---------- |
| **Distributed sync overhead**  | Early prototype in Sprint 1         | Week 1-2   |
| **Quantization accuracy loss** | Validation suite in Sprint 3        | Week 5-6   |
| **Long context cost**          | Benchmarking in Sprint 4            | Week 7-8   |
| **Multi-model interference**   | Isolation testing in Sprint 6       | Week 11-12 |
| **Documentation debt**         | Write as-you-go, review in Sprint 7 | Weeks 1-14 |

---

## PART 5: RESOURCE COSTS

### Engineering Cost Estimate

```
Team Size: 6.2 FTE + 0.9 support = 7.1 FTE

Salary Assumptions (USA Market):
├─ Senior Engineer (L5-L6): $180K-$220K/year
├─ Mid Engineer (L4): $140K-$170K/year
├─ Junior Engineer (L3): $100K-$130K/year
├─ Support roles: $80K-$120K/year
└─ Manager: $160K-$200K/year

Average: ~$145K/year per FTE
Fully-loaded (benefits, taxes, overhead): ~200K/year

Phase 3 Cost Estimate:
├─ Core Team (6.2 FTE): 6.2 × $200K × 0.33 = $410K
├─ Support (0.9 FTE): 0.9 × $150K × 0.33 = $45K
├─ Infrastructure (CI/CD, tools): $50K
├─ Contingency (10%): $50K
└─ Total: ~$550K

Conservative Estimate (USA market): $400K-$600K

Note: This assumes existing Phase 2 infrastructure, codebase, and processes.
```

### Breakdown by Component

| Component                | Engineers | Cost  | Duration |
| ------------------------ | --------- | ----- | -------- |
| **Distributed Engine**   | 1.5       | $100K | 12 weeks |
| **Quantization Stack**   | 1.5       | $120K | 8 weeks  |
| **Long Context**         | 1.0       | $80K  | 8 weeks  |
| **Fine-tuning**          | 1.0       | $80K  | 6 weeks  |
| **Model Ecosystem**      | 1.0       | $80K  | 6 weeks  |
| **Production Hardening** | 1.0       | $80K  | 6 weeks  |
| **Testing & Release**    | 1.0       | $80K  | 4 weeks  |

**Total:** 7.1 FTE, $620K

**Note:** Costs vary significantly by geography, seniority, and company structure.

---

## PART 6: SUCCESS CRITERIA & GO/NO-GO GATES

### Gate 1: End of Sprint 2 (Week 4)

**Tier 1 Features Complete**

**Go Criteria:**

- ✅ Distributed executor v1.0 working
- ✅ Continuous batching delivering 6-8× throughput
- ✅ 2-node inference working (1.8× scaling)
- ✅ 40+ integration tests passing
- ✅ 0 compiler warnings
- ✅ No critical bugs

**If No-Go:**

- Investigate root causes
- Extend Sprint 2 by 1 week
- Re-evaluate timeline

---

### Gate 2: End of Sprint 4 (Week 8)

**Tier 2 Features Complete**

**Go Criteria:**

- ✅ GPTQ + AWQ strategies complete
- ✅ Sparse attention working (32K tokens)
- ✅ KV cache compression functional
- ✅ Accuracy loss <2% on benchmarks
- ✅ 60+ integration tests passing
- ✅ Performance targets met (within 10%)

**If No-Go:**

- Identify missing requirements
- Extend by 1-2 weeks
- Reassess Tier 3 scope

---

### Gate 3: End of Sprint 6 (Week 12)

**Tier 3 Features Complete, Production Ready**

**Go Criteria:**

- ✅ Multi-model orchestration working
- ✅ 20+ models supported
- ✅ QLoRA fine-tuning complete
- ✅ Error handling comprehensive
- ✅ 80+ integration tests passing
- ✅ MTBF >1000 hours simulated

**If No-Go:**

- Identify gaps
- Extend by 2 weeks
- Plan Phase 3.5 for deferrable features

---

### Gate 4: End of Sprint 7 (Week 14)

**Release Candidate Ready**

**Go Criteria:**

- ✅ 100+ integration tests passing (>98%)
- ✅ Performance benchmarks published
- ✅ Documentation ≥95% complete
- ✅ No critical bugs
- ✅ Security review passed
- ✅ Legal/Compliance sign-off

**If No-Go:**

- Delay release by 1-2 weeks
- Fix critical issues
- Reschedule release date

---

## PART 7: STAFFING STRATEGY

### Hiring Approach

**Option 1: Use Existing Team (Lower Risk)**

- Extend Phase 2 team
- 6 engineers already understand codebase
- Onboarding minimal
- **Recommendation for Phase 3**

**Option 2: Hybrid (New + Existing)**

- Keep 3 core Phase 2 engineers
- Hire 3 new engineers specializing in:
  - Distributed systems
  - ML/Quantization
  - DevOps/Infrastructure
- **More expensive, but faster execution**

**Option 3: Outsource Specialists (Higher Risk)**

- Keep core 2 engineers
- Contract specialized teams for:
  - Distributed systems (contractor)
  - Quantization research (ML contractor)
  - DevOps (infrastructure consultant)
- **Flexible, but integration risk**

### Recommendation: **Option 1** (Use existing team)

**Rationale:**

- Phase 2 team knows codebase deeply
- Reduces onboarding overhead (2-3 weeks)
- Better code quality + fewer integration issues
- Lower total cost
- Faster execution

**Timeline:** Onboarding = 0 weeks (already familiar)

---

## PART 8: COMMUNICATION & REPORTING

### Weekly Status (Every Friday)

```
PHASE 3 WEEKLY STATUS

Week: ___ (Sprint: ___)
Status: 🟢 ON TRACK / 🟡 AT RISK / 🔴 BLOCKED

Completed:
├─ Feature A: 90% → 100% ✓
├─ Feature B: 50% → 75%
└─ Feature C: 0% → 20%

Blockers:
├─ Issue X: Investigating Y (Impact: Medium)
└─ Issue Z: Waiting for review (Impact: Low)

Next Week:
├─ Complete Feature B
├─ Start Feature C
└─ Resolve Issue X

Metrics:
├─ Test Pass Rate: 92%
├─ Compiler Warnings: 0
├─ Code Review Latency: 1.2 days
└─ Performance: On target
```

### Monthly Review (Every 4 Weeks)

```
PHASE 3 MONTHLY REVIEW

Month: ___ / Cumulative: ___ weeks

Completed (Tier 1-2 progress):
├─ Features: ___ of ___
├─ Tests: ___ passing
└─ Performance: ___ vs target

Budget Burn:
├─ Spend: $___ vs $______
├─ Variance: ___% (over/under)
└─ Projection: $_____ total

Risks & Mitigations:
├─ Risk 1: Status → Mitigation
└─ Risk 2: Status → Mitigation

Forecast:
├─ Completion: Week ___
├─ Release: Month ___
└─ Confidence: ___
```

---

## SUMMARY

**Phase 3 Resource Plan:**

| Dimension          | Value                                         |
| ------------------ | --------------------------------------------- |
| **Duration**       | 16-20 weeks (4-5 months)                      |
| **Team**           | 6.2 FTE core + 0.9 support                    |
| **Cost**           | $400K-$600K                                   |
| **Key Risk**       | Distributed sync complexity                   |
| **Critical Path**  | Distributed executor → Batching → Integration |
| **Go/No-Go Gates** | 4 major checkpoints                           |
| **Success Rate**   | >90% with proper management                   |

**Recommendation:** Plan for 20 weeks to allow buffers. Aggressive targeting of 16 weeks is possible with strong team + clear requirements.

**Next Document:** Success Criteria & Release Gates
