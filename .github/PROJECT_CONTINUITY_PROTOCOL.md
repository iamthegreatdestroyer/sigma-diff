# Project Continuity Protocol - Ryzanstein BitNet Optimization

**Version**: 1.0  
**Last Updated**: February 9, 2026  
**Protocol Owner**: @OMNISCIENT (Meta Orchestrator)  
**Status**: Active - Enforced on all sessions

---

## 1. Session Entry Protocol

### 1.1 Initial State Query (First 30 seconds of session)

```python
# Automatic on session start
CONTINUITY_CHECK = {
    "phase_state_file": ".github/workflows/phase_state.json",
    "current_phase": None,
    "current_stage": None,
    "last_session_status": None,
    "checkpoint_exists": None,
    "metrics_available": None
}
```

**Actions on entry**:

1. Read `.github/workflows/phase_state.json` → Extract current phase/stage
2. Check `/checkpoints` directory → Identify latest checkpoint
3. Scan `/reports` directory → Identify available metrics
4. Read git log → Identify last phase-tagged commit
5. Display continuity status banner

### 1.2 Continuity Status Banner

Every session displays on terminal startup:

```
╔════════════════════════════════════════════════════════════════╗
║              RYZANSTEIN PROJECT CONTINUITY STATUS              ║
╠════════════════════════════════════════════════════════════════╣
║ Current Phase: 2                                               ║
║ Current Stage: 2c                                              ║
║ Last Session: Feb 9, 2026 @ 14:32 UTC                         ║
║ Status: ✅ PHASE 2 COMPLETE - Ready for Phase 3 Setup         ║
║                                                                ║
║ Latest Checkpoint: checkpoints/phase2/optimized/latest.pt     ║
║ Metrics Available: baseline, optimized, comparison             ║
║ Last Tag: phase-2-stage-2c-20260209-143200                    ║
║                                                                ║
║ ➜ Ready to execute: Phase 3 initialization workflow            ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 2. Phase State Management

### 2.1 Phase State File Format

Location: `.github/workflows/phase_state.json`

```json
{
  "current_phase": 2,
  "current_stage": "2c",
  "status": "completed",
  "last_updated": "2026-02-09T14:32:00Z",
  "session_count": 47,
  "total_runtime_hours": 156.3,

  "completion_status": {
    "stage_2a": {
      "status": "completed",
      "duration_minutes": 45,
      "timestamp": "2026-02-09T12:00:00Z",
      "checkpoint": "checkpoints/phase2/baseline/latest.pt"
    },
    "stage_2b": {
      "status": "completed",
      "duration_minutes": 48,
      "timestamp": "2026-02-09T13:00:00Z",
      "checkpoint": "checkpoints/phase2/optimized/latest.pt"
    },
    "stage_2c": {
      "status": "completed",
      "duration_minutes": 20,
      "timestamp": "2026-02-09T14:32:00Z",
      "metrics": "reports/comparison_results.json"
    }
  },

  "phase_exit_checkpoint": {
    "commit_hash": "a1b2c3d4e5f6g7h8",
    "branch": "sprint6/api-integration",
    "tag": "phase-2-stage-2c-20260209-143200",
    "documentation": "DELIVERABLES_SUMMARY_PHASE2.md"
  },

  "next_phase": {
    "phase": 3,
    "expected_stages": ["3a", "3b", "3c"],
    "estimated_start": "2026-02-10T10:00:00Z",
    "setup_ready": false
  },

  "metrics_summary": {
    "baseline_loss": 2.1847,
    "optimized_loss": 1.9234,
    "loss_improvement_percent": 11.9,
    "inference_speedup": 1.43,
    "memory_reduction_percent": 18.5
  }
}
```

### 2.2 Phase State Updates

Phase state is updated automatically at:

- ✅ **Stage completion** - After each stage passes validation
- ✅ **Phase completion** - After final stage + comparative analysis
- ✅ **Checkpoint creation** - When model saved
- ✅ **Metrics generation** - When validation metrics available
- ✅ **Error/failure** - When rollback required

---

## 3. Checkpoint Management

### 3.1 Checkpoint Directory Structure

```
checkpoints/
├── phase1/
│   ├── baseline/
│   │   ├── epoch_00.pt
│   │   ├── epoch_05.pt
│   │   └── latest.pt → epoch_10.pt
│   └── optimized/
│       ├── epoch_00.pt
│       ├── epoch_05.pt
│       └── latest.pt → epoch_10.pt
├── phase2/
│   ├── baseline/
│   │   ├── epoch_00.pt
│   │   ├── epoch_05.pt
│   │   └── latest.pt → epoch_10.pt
│   └── optimized/
│       ├── epoch_00.pt
│       ├── epoch_05.pt
│       └── latest.pt → epoch_10.pt
└── recovery/
    └── [auto-saved checkpoints for failed runs]
```

### 3.2 Checkpoint Loading Logic

```python
def load_session_checkpoint():
    """
    Automatic checkpoint selection on session entry
    Priority: Latest optimized > Latest baseline > Recovery > Start fresh
    """
    checkpoint_priority = [
        Path("checkpoints/phase2/optimized/latest.pt"),
        Path("checkpoints/phase2/baseline/latest.pt"),
        Path("checkpoints/recovery/latest.pt"),
    ]

    for checkpoint in checkpoint_priority:
        if checkpoint.exists():
            return load_checkpoint(checkpoint)

    return initialize_fresh_model()
```

### 3.3 Checkpoint Retention Policy

- **Latest 2 per stage**: Keep recent checkpoints for continuity
- **Phase exit checkpoint**: Always retained (tagged + backed up)
- **Recovery checkpoints**: Kept for 7 days
- **Full cleanup**: On successful phase completion + archive

---

## 4. Metrics & Documentation Continuity

### 4.1 Metrics Capture & Archive

Each phase automatically captures:

```
reports/
├── phase1/
│   ├── baseline/metrics.json
│   ├── optimized/metrics.json
│   └── PHASE1_ANALYSIS.md
├── phase2/
│   ├── baseline/metrics.json
│   ├── optimized/metrics.json
│   ├── comparison_results.json
│   └── PHASE2_ANALYSIS.md
├── deliverables_index.json
└── metrics_archive/ [compressed]
    └── phase2_20260209.tar.gz
```

### 4.2 Documentation Auto-Update

On phase completion, the following are auto-generated:

- `DELIVERABLES_SUMMARY_PHASEXY.md` - Phase deliverables
- `DELIVERABLES_INDEX.md` - Updated index
- Phase completion entry in `CHANGELOG.md`
- `progress_timeline.json` - Session-by-session timeline

---

## 5. Git Integration & Tagging

### 5.1 Automatic Phase Tagging

Tags created on phase completion:

```
Format: phase-{N}-stage-{NX}-{YYYYMMDD-HHMMSS}

Examples:
- phase-2-stage-2c-20260209-143200
- phase-3-stage-3a-20260210-100000
```

### 5.2 Phase Commit Messages

Standard format enforced:

```
chore(phase-2-stage-2c): Complete comparative validation

- ✅ Baseline training: 10 epochs completed
- ✅ Optimized training: 10 epochs completed
- ✅ Inference comparison: TTFT 1.43x speedup, 18.5% memory reduction
- 📊 Metrics: baseline_loss=2.1847, optimized_loss=1.9234
- 📈 Improvement: 11.9% loss reduction

Metrics:
- File: reports/PHASE2_ANALYSIS.md
- Checkpoint: checkpoints/phase2/optimized/latest.pt

[phase-2-stage-2c]
[CONC-101] [sprint6/api-integration]
```

### 5.3 Git Workflow Rules

```
Branch naming: sprint{N}/{feature-name}
  Example: sprint6/api-integration

Tag naming: phase-{N}-stage-{NX}-{timestamp}
  Example: phase-2-stage-2c-20260209-143200

PR titles: "Phase {N} Stage {NX}: [Description]"
  Example: "Phase 2 Stage 2c: Complete Comparative Validation"
```

---

## 6. Session Handoff Protocol

### 6.1 End-of-Session Checklist

Before session exit, automation ensures:

- [ ] Phase state file updated
- [ ] Latest checkpoint saved + symlinked
- [ ] Metrics exported to JSON
- [ ] Documentation generated
- [ ] Git changes committed + tagged (if phase complete)
- [ ] Continuity banner logged to session file
- [ ] Next session guidance written to `NEXT_SESSION.md`

### 6.2 NEXT_SESSION.md Generation

Automatically created at session end:

````markdown
# Next Session: Quick Start Guide

**Last Session**: February 9, 2026 @ 14:32 UTC

## Current Status

- Phase: 2
- Stage: 2c (COMPLETE ✅)
- Status: Ready for Phase 3

## Latest Checkpoint

- Path: `checkpoints/phase2/optimized/latest.pt`
- Epoch: 10
- Loss: 1.9234

## Available Metrics

- Baseline metrics: `reports/phase2/baseline/metrics.json`
- Optimized metrics: `reports/phase2/optimized/metrics.json`
- Comparison: `reports/PHASE2_ANALYSIS.md`

## Recommended Next Action

Execute: `python -m workflows.task_automation --workflow-type phase-transition-setup`

This will:

1. Initialize Phase 3 directories
2. Create Phase 3 configuration
3. Update phase state to Phase 3a
4. Ready system for API Integration

## Commands to Resume

```bash
# Activate environment
. .venv/Scripts/Activate.ps1

# View current status
python -c "import json; print(json.dumps(json.load(open('.github/workflows/phase_state.json')), indent=2))"

# Start Phase 3 setup
python .github/workflows/task_automation.py --workflow-type phase-transition-setup
```
````

---

## 7. Failure & Recovery Protocol

### 7.1 Automatic Failure Detection

Monitored failure modes:

```
FAILURE_TRIGGERS = {
    "cuda_out_of_memory": {"action": "rollback", "checkpoint": "recovery"},
    "nans_in_loss": {"action": "rollback", "checkpoint": "recovery"},
    "training_stalled": {"action": "inspect", "checkpoint": "current"},
    "inference_segfault": {"action": "rollback", "checkpoint": "recovery"},
    "ci_pipeline_failure": {"action": "alert", "checkpoint": "last_valid"}
}
```

### 7.2 Recovery Procedures

On failure:

1. **Automatic checkpoint save** to `checkpoints/recovery/`
2. **State snapshot** to `recovery_state.json`
3. **Error logging** to `logs/recovery/`
4. **Rollback attempt** to last valid checkpoint (max 3 retries)
5. **Alert generation** with root cause analysis
6. **Manual intervention** guidance after failed retries

### 7.3 Manual Recovery Commands

```bash
# Inspect recovery checkpoint
python -c "
import torch
ckpt = torch.load('checkpoints/recovery/latest.pt')
print('Available keys:', ckpt.keys())
print('Model state dict keys:', len(ckpt['model_state_dict']))
"

# Retry last failed stage
python .github/workflows/task_automation.py --workflow-type comparative-validation-cycle --resume-from recovery

# Roll back to last known good state
git checkout phase-2-stage-2c-20260209-143200
python scripts/reload_checkpoint.py --checkpoint checkpoints/phase2/optimized/latest.pt
```

---

## 8. Session Analytics & Reporting

### 8.1 Session Metrics Tracked

```json
{
  "session_number": 47,
  "start_time": "2026-02-09T12:00:00Z",
  "end_time": "2026-02-09T14:32:00Z",
  "duration_minutes": 152,
  "phase": 2,
  "stages_executed": ["2a", "2b", "2c"],
  "stages_successful": ["2a", "2b", "2c"],
  "checkpoints_saved": 3,
  "metrics_generated": 4,
  "git_operations": {
    "commits": 3,
    "tags": 1,
    "pushes": 1
  },
  "continuity_score": 0.98,
  "handoff_status": "ready"
}
```

### 8.2 Continuity Score Calculation

```python
continuity_score = (
    (phases_completed / phases_attempted * 0.4) +
    (checkpoints_valid / checkpoints_expected * 0.3) +
    (metrics_captured / metrics_expected * 0.2) +
    (git_operations_clean / git_operations_total * 0.1)
)
# 0.0 - 1.0 scale; target: > 0.95
```

---

## 9. Multi-Agent Continuity Coordination

### 9.1 Agent-Specific Checkpoints

Each agent logs continuity state:

- **@ARCHITECT**: Design decisions, ADRs, phase architecture
- **@APEX**: Code quality checkpoints, refactoring history
- **@VELOCITY**: Performance baselines, optimization tracking
- **@TENSOR**: Model architecture snapshots, training curves
- **@FLUX**: Deployment state, CI/CD status
- **@OMNISCIENT**: Orchestration state, cross-agent coordination

### 9.2 Cross-Agent Continuity

```json
{
  "agent_states": {
    "@ARCHITECT": {
      "last_active": "2026-02-09T13:45:00Z",
      "phase": 2,
      "decisions_documented": 8
    },
    "@APEX": {
      "last_active": "2026-02-09T13:30:00Z",
      "phase": 2,
      "code_quality_score": 0.92
    },
    "@TENSOR": {
      "last_active": "2026-02-09T14:15:00Z",
      "phase": 2,
      "model_checkpoint": "checkpoints/phase2/optimized/latest.pt"
    }
  }
}
```

---

## 10. Enforcement & Auditing

### 10.1 Continuity Audit Trail

Every operation logged to `logs/continuity_audit.log`:

```
[2026-02-09 14:32:00] INFO: Session 47 ended successfully
[2026-02-09 14:32:01] INFO: Phase state updated: phase=2, stage=2c
[2026-02-09 14:32:02] INFO: Checkpoint saved: checkpoints/phase2/optimized/latest.pt
[2026-02-09 14:32:03] INFO: Metrics captured: 4 files
[2026-02-09 14:32:04] INFO: Git operations: 3 commits, 1 tag
[2026-02-09 14:32:05] INFO: Continuity score: 0.98
[2026-02-09 14:32:06] INFO: Handoff status: READY for next session
```

### 10.2 Continuity Violations

Automatic alerts on:

- Checkpoint not saved at stage end
- Phase state file > 1 hour out of sync
- Metrics directory missing expected files
- Git operations not properly tagged
- Session exit without proper handoff

---

## 11. Quick Reference Commands

```bash
# Check continuity status
python -c "import json; print(json.load(open('.github/workflows/phase_state.json')))"

# List all checkpoints with status
python scripts/checkpoint_status.py

# Generate continuity report
python scripts/generate_continuity_report.py

# Verify session handoff
python scripts/verify_handoff.py

# Manual phase transition
python .github/workflows/task_automation.py --workflow-type phase-transition-setup

# Recovery mode (if needed)
python scripts/recovery_mode.py --checkpoint recovery
```

---

**Next Review Date**: February 16, 2026  
**Protocol Maintainer**: @OMNISCIENT  
**Last Verified**: February 9, 2026, 14:32 UTC
