# OMNISCIENT Proactive Engagement Triggers

**Version**: 1.0  
**Effective**: February 9, 2026  
**Orchestrator**: @OMNISCIENT (Meta-Agent)  
**Status**: Active - Continuous monitoring

---

## Overview

This document defines the automatic engagement triggers that enable @OMNISCIENT to operate proactively rather than reactively. Rather than waiting for user invocation, @OMNISCIENT monitors workspace state and automatically routes work to optimal agent teams.

**Trigger Philosophy**:

- Detect state changes automatically
- Route to optimal agent(s) without user intervention
- Maximize autonomy within clear boundaries
- Log all triggered actions for audit

---

## 1. Comparative Validation Auto-Trigger

### Trigger Condition

**File Change**: `training_configuration.yaml` modified with new training args + no baseline benchmark exists

```python
TRIGGER = {
    "type": "training_config_change",
    "detector": "file_modification_watcher",
    "watch_paths": ["training_configuration.yaml"],
    "conditions": [
        "config file changed",
        "new training parameters detected",
        "baseline_metrics.json NOT in reports/"
    ],
    "action": "ENGAGE_COMPARATIVE_VALIDATION",
    "priority": "HIGH"
}
```

### Engagement Logic

```python
if file_changed("training_configuration.yaml"):
    config_new = parse_yaml("training_configuration.yaml")
    config_old = cached_config

    if config_different(config_new, config_old):
        baseline_exists = file_exists("reports/baseline/metrics.json")
        optimized_exists = file_exists("reports/optimized/metrics.json")

        if not (baseline_exists and optimized_exists):
            trigger_workflow("comparative-validation-cycle")
            log_action("AUTO_TRIGGERED: comparative-validation-cycle")
```

### Agent Team Routing

```
Primary: @APEX (code execution, training setup)
Support:
├─ @VELOCITY (performance monitoring)
├─ @AXIOM (complexity analysis)
├─ @ECLIPSE (validation & testing)
└─ @TENSOR (model optimization insights)
```

### Engagement Format

```
📊 AUTO-TRIGGER: Comparative Validation Cycle
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Config changed: training_configuration.yaml
New params: learning_rate=0.0005, warmup_steps=1000

Missing baseline comparison. Auto-triggering workflow:
  → Phase 2a: Baseline training (NO optimizations)
  → Phase 2b: Optimized training (WITH Phase 1)
  → Phase 2c: Comparative inference analysis

Estimated duration: 2 hours 15 minutes
Expected completion: 16:45 UTC

Agent team: @APEX → @VELOCITY → @ECLIPSE
Status: [████████░░] Starting Phase 2a...
```

---

## 2. Performance Regression Detection

### Trigger Condition

**Metrics Change**: New metric file shows > 5% performance degradation vs previous best

```python
TRIGGER = {
    "type": "performance_regression",
    "detector": "metrics_comparator",
    "watch_paths": ["reports/**/metrics.json"],
    "threshold": 0.05,  # 5% degradation
    "conditions": [
        "new metrics captured",
        "current_metric < (best_metric * 0.95)",
        "degradation not expected per release notes"
    ],
    "action": "ALERT_AND_INVESTIGATE",
    "priority": "CRITICAL"
}
```

### Regression Analysis Logic

```python
def detect_regression(new_metrics):
    best_metrics = load_best_metrics()

    regression_detected = {
        "loss_regression": new_metrics['loss'] > best_metrics['loss'] * 1.05,
        "inference_slowdown": new_metrics['ttft'] > best_metrics['ttft'] * 1.05,
        "memory_increase": new_metrics['memory'] > best_metrics['memory'] * 1.05
    }

    if any(regression_detected.values()):
        return trigger_investigation(regression_detected)
```

### Engagement on Regression

```
🚨 CRITICAL: Performance Regression Detected
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Previous best: loss=1.9234, TTFT=142ms, memory=4.2GB
Current run:   loss=2.0198, TTFT=156ms, memory=5.1GB

Regression metrics:
  ❌ Loss: +5.0% degradation (THRESHOLD: 5%)
  ❌ TTFT: +9.9% slowdown (THRESHOLD: 5%)
  ⚠️  Memory: +21.4% increase (EXCEEDS: 5%)

Root cause analysis agents assigned:
  @VELOCITY: Performance profiling
  @APEX: Code diff analysis
  @FORTRESS: Resource constraint check
  @TENSOR: Model state analysis

Stopping training to investigate.
```

### Engagement Actions

1. **Pause execution** - Stop current training immediately
2. **Capture state** - Save full checkpoint + metrics snapshot
3. **Trigger investigation** - @VELOCITY + @APEX root cause analysis
4. **Generate report** - Automatic regression analysis report
5. **Suggest rollback** - Option to revert to last known good

---

## 3. Phase Completion Auto-Trigger

### Trigger Condition

**Stage Status**: Git log shows phase-tagged commit + phase_state.json indicates stage complete

```python
TRIGGER = {
    "type": "phase_completion",
    "detector": "git_tag_watcher",
    "watch_pattern": "phase-{N}-stage-{NX}-.*",
    "conditions": [
        "git tag matching pattern created",
        "phase_state.json updated",
        "all stage metrics available",
        "comparative analysis complete"
    ],
    "action": "AUTO_TRIGGER_NEXT_PHASE_SETUP",
    "priority": "HIGH"
}
```

### Phase Completion Detection

```python
def detect_phase_completion():
    latest_tag = get_latest_git_tag()

    if match_pattern(latest_tag, "phase-{N}-stage-{NX}-.*"):
        phase_state = load_phase_state()

        if phase_state['status'] == 'completed':
            # Verify all stage files exist
            metrics_files = [
                f"reports/phase{N}/{stage}/metrics.json"
                for stage in completed_stages
            ]

            if all(file_exists(f) for f in metrics_files):
                trigger_next_phase_setup(phase_state)
```

### Auto-Trigger Next Phase Setup

```
✅ PHASE 2 COMPLETE: Triggering Phase 3 Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 2 Summary:
  ✅ Stage 2a: Baseline training completed
  ✅ Stage 2b: Optimized training completed
  ✅ Stage 2c: Comparative analysis completed

Metrics Summary:
  Baseline Loss: 2.1847
  Optimized Loss: 1.9234
  Loss Improvement: 11.9%
  Speedup: 1.43x
  Memory Reduction: 18.5%

Initializing Phase 3: API Integration & Distributed Inference

Agent team assigned:
  @ARCHITECT: Phase 3 architecture
  @SYNAPSE: API integration design
  @FLUX: Distributed inference setup
  @ECLIPSE: Integration testing

Directories created: checkpoints/phase3, reports/phase3, logs/phase3
Config generated: training_configuration_phase3.yaml
Status: Ready for Phase 3a kickoff

Next action: Manual review + Phase 3a launch authorization
```

---

## 4. CI/CD Failure Auto-Investigation

### Trigger Condition

**CI Pipeline**: GitHub Actions workflow failure detected in training_ci.yml OR desktop-build.yml

```python
TRIGGER = {
    "type": "ci_failure",
    "detector": "github_actions_watcher",
    "watch_workflows": ["training_ci.yml", "desktop-build.yml", "ci.yml"],
    "conditions": [
        "workflow run failed",
        "exit code != 0",
        "failure logs available"
    ],
    "action": "AUTO_INVESTIGATE_AND_REPORT",
    "priority": "CRITICAL"
}
```

### CI Failure Investigation

```python
def investigate_ci_failure(workflow_run):
    logs = fetch_workflow_logs(workflow_run)

    # Analyze failure type
    failure_type = classify_failure(logs)

    if failure_type == "cuda_error":
        trigger_agents(["@CORE", "@VELOCITY", "@FORTRESS"])
    elif failure_type == "import_error":
        trigger_agents(["@APEX", "@ECLIPSE"])
    elif failure_type == "timeout":
        trigger_agents(["@VELOCITY", "@AXIOM"])
    else:
        trigger_agents(["@OMNISCIENT"])  # Unknown - escalate to meta
```

### Engagement on CI Failure

```
🚨 CI FAILURE DETECTED: training_ci.yml run #89
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Failure type: CUDA out of memory
Job: train-model (GPU: A100)

Error log excerpt:
  RuntimeError: CUDA out of memory. Tried to allocate 2048MB
  When allocating: memory pool on device 0

Likely causes (ranked by probability):
  1. Batch size too large (32) for model size (256 embedding × 4 heads)
  2. Memory fragmentation from previous runs
  3. GPU driver memory leak

Investigation agents:
  @CORE: Memory allocation patterns
  @VELOCITY: Batch size optimization
  @FORTRESS: Resource constraints

Suggested fixes:
  1. Reduce batch_size: 32 → 16 in training_configuration.yaml
  2. Enable torch.cuda.empty_cache() between epochs
  3. Verify GPU memory after job cleanup

Automatic actions:
  ✓ Saved logs: logs/ci_failures/run89_analysis.json
  ✓ Generated fix suggestions: SUGGESTED_CI_FIXES.md
  ✓ Reduced batch size in config (pending manual approval)
  ✓ Tagged commit: ci-failure-run89-investigation

Manual approval needed: Apply batch size reduction + retry?
```

---

## 5. Documentation Gap Detection

### Trigger Condition

**Documentation State**: Code changed but no corresponding documentation update

```python
TRIGGER = {
    "type": "documentation_gap",
    "detector": "git_diff_analyzer",
    "conditions": [
        "python files committed",
        "yaml config files changed",
        "no docs/* files in same commit",
        "no .md files in PR"
    ],
    "action": "REMIND_AND_SUGGEST_DOCUMENTATION",
    "priority": "MEDIUM"
}
```

### Documentation Gap Detection

```python
def detect_doc_gap(commit_files):
    code_files = [f for f in commit_files if is_code_file(f)]
    doc_files = [f for f in commit_files if is_doc_file(f)]

    if code_files and not doc_files:
        doc_gap_severity = estimate_severity(code_files)

        if doc_gap_severity > 0.5:
            trigger_documentation_reminder(code_files)
```

### Engagement on Documentation Gap

```
📝 DOCUMENTATION REMINDER: Code changes detected
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Files changed:
  - RYZEN-LLM/scripts/training_loop.py (+47, -12)
  - training_configuration.yaml (modified)

No documentation updates found in this PR.

Suggested documentation:
  @SCRIBE: Update training_loop.py docstrings
  @VANGUARD: Update TRAINING_GUIDE.md
  @MENTOR: Code review for API changes

Files needing update:
  ❌ TRAINING_GUIDE.md (last updated: 2 weeks ago)
  ❌ RYZEN-LLM/scripts/training_loop.py (docstrings outdated)
  ❌ CHANGELOG.md (not updated)

Auto-comment on PR:
  "👋 Reminder: Please update documentation for API changes.
   Suggested files: TRAINING_GUIDE.md, docstrings, CHANGELOG.md"

Assistance available: Type @SCRIBE generate-docs [file]
```

---

## 6. Uncommitted Changes Warning

### Trigger Condition

**Workspace State**: Significant changes in workspace without corresponding git commits

```python
TRIGGER = {
    "type": "uncommitted_changes",
    "detector": "git_status_watcher",
    "threshold_files": 5,
    "threshold_lines": 100,
    "interval": 300,  # Check every 5 minutes
    "conditions": [
        "unstaged changes > threshold",
        "no commit for > 30 minutes",
        "no .git ignore patterns match files"
    ],
    "action": "REMIND_AND_SUGGEST_COMMIT",
    "priority": "LOW"
}
```

### Engagement on Uncommitted Changes

```
💾 REMINDER: Uncommitted changes detected
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changed files: 8
Lines modified: +156, -42
Time since last commit: 47 minutes

Files awaiting commit:
  M  RYZEN-LLM/scripts/training_loop.py (+45, -12)
  M  training_configuration.yaml (+8, -2)
  M  ARCHITECTURE.md (+103, -28)
  ?? debug_run_training.py (new file)

Suggested commit:
  git add .
  git commit -m "feat: Phase 2b - Optimize training with kernel improvements

  - Implement semantic compression in training loop
  - Add RLVR inference scaling configuration
  - Update architecture documentation

  [sprint6/api-integration]"

Agent assistance: Type @APEX generate-commit-msg [files]
```

---

## 7. Checkpoint Staleness Detection

### Trigger Condition

**Checkpoint State**: Latest checkpoint > 2 hours old and training still running

```python
TRIGGER = {
    "type": "checkpoint_staleness",
    "detector": "checkpoint_age_monitor",
    "max_age_minutes": 120,
    "conditions": [
        "time_since_checkpoint > max_age",
        "training_process_running",
        "no_new_checkpoint_created"
    ],
    "action": "TRIGGER_MANUAL_CHECKPOINT_SAVE",
    "priority": "MEDIUM"
}
```

### Engagement on Stale Checkpoints

```
⚠️  WARNING: Checkpoint is stale (145 min old)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Last checkpoint: 143 minutes ago
Training status: RUNNING (epoch 7/10)

Risk: Loss of progress if process crashes

Recommended action:
  1. Continue training (automatic save at epoch end)
  2. Manual checkpoint: python save_checkpoint.py
  3. Review checkpoint interval in config

Current checkpoint interval: 100 steps (~45 min)
Next auto-checkpoint: ~12 minutes

Auto-action: Created reminder for manual save if no auto-save in 15 min
```

---

## 8. Test Coverage Regression Detection

### Trigger Condition

**Test State**: Test failures after recent commits

```python
TRIGGER = {
    "type": "test_coverage_regression",
    "detector": "ci_test_result_analyzer",
    "watch_patterns": ["test_*.py", "tests/"],
    "conditions": [
        "test_run_failed",
        "coverage < previous_coverage",
        "failure not in known_flaky_tests"
    ],
    "action": "TRIGGER_TEST_DEBUG_WORKFLOW",
    "priority": "HIGH"
}
```

### Test Failure Investigation

```
❌ TEST FAILURE DETECTED: test_model_inference.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test: test_inference_output_shape
Status: FAILED
Error: AssertionError: (1, 10) != (1, 12)

Related commits:
  - 3a9f8c: Add positional encoding to transformer
  - 7b2c9d: Update embedding dimension to 256

Most likely cause: Output shape changed due to embedding changes

Investigation agents:
  @APEX: Code diff analysis
  @ECLIPSE: Test update recommendations
  @TENSOR: Model shape validation

Suggested fix:
  Update test expected shape: (1, 10) → (1, 12)
  OR: Revert embedding dimension change

Auto-actions:
  ✓ Created issue: #CONC-102 test_inference_output_shape failure
  ✓ Assigned to: @APEX, @ECLIPSE
  ✓ Generated fix suggestions: TEST_FIX_SUGGESTIONS.md
```

---

## 9. Integration Failure Auto-Escalation

### Trigger Condition

**Integration**: API endpoint failures OR inference validation failures

```python
TRIGGER = {
    "type": "integration_failure",
    "detector": "integration_test_monitor",
    "conditions": [
        "api_endpoint_error OR inference_validation_error",
        "error_rate > 0.05",  # 5% error threshold
        "error_persists > 5 minutes"
    ],
    "action": "AUTO_ESCALATE_TO_ARCHITECTURE",
    "priority": "CRITICAL"
}
```

### Engagement on Integration Failure

```
🚨 INTEGRATION FAILURE: API endpoint /inference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Error: Connection timeout to model backend
Status: Persistent (8 consecutive failures)

Affected components:
  - REST API endpoint: /api/v1/inference
  - Backend worker: inference-worker-1
  - Model service: spell-checker-v2

Investigation escalation:
  Level 1 (Local): @APEX, @SYNAPSE
  Level 2 (Architecture): @ARCHITECT, @FLUX
  Level 3 (Emergency): @OMNISCIENT

Actions taken:
  ✓ Marked endpoint unhealthy
  ✓ Rerouted traffic to backup endpoint
  ✓ Captured failure logs
  ✓ Triggered architecture review

Fallback status: Running on checkpoint from 30 minutes ago
Expected recovery: Awaiting root cause diagnosis
```

---

## 10. Threshold-Based Auto-Triggers

### 10.1 Training Loss Explosion

```python
TRIGGER = {
    "type": "loss_explosion",
    "condition": "current_loss > (baseline_loss * 10)",
    "action": "IMMEDIATE_TRAINING_HALT",
    "agents": ["@VELOCITY", "@TENSOR", "@APEX"],
    "priority": "CRITICAL"
}
```

### 10.2 Memory Leak Detection

```python
TRIGGER = {
    "type": "memory_leak",
    "condition": "memory_used increases by >2% per training step",
    "action": "TRIGGER_MEMORY_PROFILING",
    "agents": ["@VELOCITY", "@CORE", "@ECLIPSE"],
    "priority": "HIGH"
}
```

### 10.3 GPU Utilization Too Low

```python
TRIGGER = {
    "type": "low_gpu_utilization",
    "condition": "avg_gpu_utilization < 40% for > 10 minutes",
    "action": "OPTIMIZE_BATCH_SIZE_AND_CONFIG",
    "agents": ["@VELOCITY", "@APEX"],
    "priority": "MEDIUM"
}
```

---

## 11. Manual Trigger Override

Users can manually trigger investigations:

```bash
# Trigger comparative validation immediately
python -m workflows.engagement_triggers --trigger comparative-validation

# Force regression investigation
python -m workflows.engagement_triggers --trigger regression-investigation

# Manual phase transition
python -m workflows.engagement_triggers --trigger phase-transition --phase 3

# Comprehensive health check
python -m workflows.engagement_triggers --trigger full-health-check

# Specific agent engagement
python -m workflows.engagement_triggers --trigger specific-agent --agent @VELOCITY --task optimization-audit
```

---

## 12. Engagement Audit Trail

All auto-triggered engagements logged to `logs/engagement_audit.log`:

```
[2026-02-09 15:45:32] TRIGGER: comparative_validation
  Reason: config change detected in training_configuration.yaml
  Agents: @APEX, @VELOCITY, @ECLIPSE, @TENSOR
  Status: ACTIVATED

[2026-02-09 16:12:45] TRIGGER: performance_regression
  Reason: loss degradation 5.0% detected
  Baseline: 1.9234, Current: 2.0198
  Priority: CRITICAL
  Status: INVESTIGATING

[2026-02-09 17:20:00] TRIGGER: phase_completion
  Reason: phase-2-stage-2c-20260209-143200 tag created
  Next action: Phase 3 setup
  Status: ACTIVATED
```

---

## 13. Disabling Triggers (Emergency Mode)

In case of excessive triggers or emergency:

```bash
# Disable all auto-triggers
export OMNISCIENT_AUTO_TRIGGERS=disabled

# Disable specific trigger
export OMNISCIENT_DISABLE_TRIGGERS=performance_regression,ci_failure

# Resume triggers
export OMNISCIENT_AUTO_TRIGGERS=enabled
```

---

**Last Updated**: February 9, 2026, 14:32 UTC  
**Maintained By**: @OMNISCIENT  
**Review Schedule**: Weekly  
**Emergency Contact**: @APEX, @ARCHITECT
