# Phase 3a Training Monitoring & Orchestration

Complete toolkit for monitoring, tracking, and alerting on Phase 3a training execution.

## 📋 Overview

**Phase 3a Training**: Validates Phase 1 optimization stack on scaled model (9.5M parameters)

**Monitoring Tools**:

- `monitor_phase3a_training.py` - Real-time progress tracking with auto-completion
- `status_checker_phase3a.py` - Lightweight status snapshots with resource monitoring
- `alert_service_phase3a.py` - Multi-channel completion notifications
- `orchestrator_phase3a.py` - Unified orchestration of all components

**Current Status**: Training ACTIVE (Baseline ✅ COMPLETE, Optimized ⏳ IN PROGRESS)

---

## 🚀 Quick Start

### Option 1: Unified Orchestrator (RECOMMENDED) 🎯

Start everything together with automatic orchestration:

```powershell
# Full orchestration - starts monitor, checker, waits for completion, triggers alerts
python orchestrator_phase3a.py --full

# Or just wait for completion (monitor background training)
python orchestrator_phase3a.py --wait
```

**What it does**:

- ✅ Starts progress monitor (real-time tracking)
- ✅ Starts periodic status checker (background monitoring)
- ✅ Waits up to 20 minutes for completion
- ✅ Auto-triggers alerts when done
- ✅ Displays results with speedup validation

**Output**:

- Console: Real-time progress + alerts
- Files: orchestrator.log, monitor.log, phase3a_status.json, alerts.log
- Results: phase3_stage3a_comparison.json

---

### Option 2: Individual Tools

If you prefer fine-grained control:

#### Monitor Only (Real-Time Progress)

```powershell
python monitor_phase3a_training.py
```

- Displays epoch/loss every 30 seconds
- Auto-exits on completion
- Auto-triggers alerts
- **Best for**: Attentive monitoring with detailed visibility

#### Status Checker Only (Lightweight Checks)

```powershell
# Single check + exit
python status_checker_phase3a.py --once

# Wait for completion (blocks until done)
python status_checker_phase3a.py --wait

# Periodic checks (every 30 seconds, runs for 20 min)
python status_checker_phase3a.py --periodic 30

# Verbose output
python status_checker_phase3a.py --once --verbose
```

- **Best for**: Flexible polling, unattended monitoring, integration with other tools

#### Alert Service Only (Manual Triggering)

```powershell
# Trigger with detailed results
python alert_service_phase3a.py --detailed

# Quiet mode
python alert_service_phase3a.py --quiet

# Console only (no system notifications)
python alert_service_phase3a.py --no-system
```

- **Best for**: Manual result review, integration workflows

---

## 📊 Tool Details

### 1. Orchestrator (`orchestrator_phase3a.py`)

**The master control panel** - Coordinates all monitoring components.

#### Features

- Single point of control for all monitoring
- Unified logging across components
- Automatic workflows (full orchestration mode)
- Component lifecycle management
- Output monitoring and aggregation

#### Commands

```powershell
# RECOMMENDED - Full workflow (20 min max)
python orchestrator_phase3a.py --full

# Just wait for completion (useful for running in background)
python orchestrator_phase3a.py --wait

# Start individual components
python orchestrator_phase3a.py --monitor        # Monitor only
python orchestrator_phase3a.py --checker        # Checker only
python orchestrator_phase3a.py --alert          # Alerts only
```

#### Output Files

- `orchestrator.log` - Master orchestration log
- `monitor.log` - Monitor component output
- `alerts.log` - Alert system events
- `phase3a_status.json` - Status snapshots
- `phase3_stage3a_comparison.json` - Final results

---

### 2. Monitor (`monitor_phase3a_training.py`)

**Real-time progress tracking with intelligent completion detection.**

#### Features

- Detects running Python process
- Parses epoch/loss/throughput in real-time
- Monitors file creation for completion
- Auto-detects training completion
- Loads and displays results
- Multi-format output (table, json)

#### Usage

```powershell
python monitor_phase3a_training.py
```

#### Output Example

```
📊 PHASE 3a TRAINING MONITOR
================================================================================
Checking interval: 10 seconds
Max duration: 900 seconds (15 minutes)

[10:15:30] ℹ️  Process detected - Training in progress...
[10:15:30] 📈 Process: 1234 | Memory: 2,145 MB | CPU: 45% | Elapsed: 285s
[10:15:30] 📊 Status: BASELINE PHASE COMPLETE - OPTIMIZED RUNNING (Epoch 8/10)

TRAINING PROGRESS:
├─ Latest Epoch: 8/10
├─ Latest Loss: 0.3767
├─ Latest Throughput: 11.2 tok/s
├─ Timestamp: 10:15:28
└─ Process Status: Running

FILES DETECTED:
├─ Total Checkpoints: 10 files
├─ Latest: scaled_model_epoch_8.pt (412 MB)
├─ Comparison JSON: Not yet (expected at completion)
└─ Last Modified: 10:15:25

ELAPSED TIME: 285 seconds

================================================================================
[10:15:40] ℹ️  Process detected - Training in progress...
```

#### When Completion Detected

```
🎉 COMPLETION DETECTED!

=================== 📊 TRAINING RESULTS ===================
Baseline Training:
  • Total Time: 412.3 seconds
  • Final Loss: 6.21
  • Throughput: 11.4 tokens/sec

Optimized Training:
  • Total Time: 289.7 seconds
  • Final Loss: 0.38
  • Throughput: 11.8 tokens/sec

SPEEDUP ANALYSIS:
  • Improvement: 412.3 / 289.7 = 1.42x faster
  • Speedup %: 42.3%
  • ✅ Target Achieved: YES (≥25% required)

Alerts triggered on completion.
Details: Check phase3_stage3a_comparison.json
```

---

### 3. Status Checker (`status_checker_phase3a.py`)

**Lightweight status snapshots with resource monitoring.**

#### Features

- Process info (memory, CPU %)
- File statistics
- Checkpoint counting
- JSON state export
- Multiple execution modes
- Event-driven or periodic polling

#### Usage

```powershell
# Single check + exit (shows status once)
python status_checker_phase3a.py --once

# Block until completion (up to 20 min timeout)
python status_checker_phase3a.py --wait

# Periodic checks (check every N seconds)
python status_checker_phase3a.py --periodic 30

# Verbose output (detailed information)
python status_checker_phase3a.py --once --verbose

# Combined
python status_checker_phase3a.py --periodic 60 --verbose
```

#### Output Example

```
PHASE 3a STATUS CHECK
================================================================================
Check Time: 2025-03-14 10:15:35 UTC
Training PID: 1234

PROCESS INFO:
  Running: YES ✅
  Memory: 2,145 MB
  CPU: 45.2%
  Status: Training in progress

FILE STATISTICS:
  Checkpoints: 10 files
  Latest checkpoint: scaled_model_epoch_8.pt (412 MB, modified 30s ago)
  Has comparison: NO (expected at completion)

QUICK STATUS: RUNNING at Epoch 8/10, Memory: 2.1 GB, CPU: 45%
  Next check in: 30 seconds
================================================================================
```

#### State File Export

```json
{
  "timestamp": "2025-03-14T10:15:35Z",
  "running": true,
  "pid": 1234,
  "memory_mb": 2145,
  "cpu_percent": 45.2,
  "checkpoint_files": 10,
  "latest_checkpoint": "scaled_model_epoch_8.pt",
  "latest_checkpoint_size_mb": 412,
  "has_comparison_json": false,
  "elapsed_seconds": 285
}
```

---

### 4. Alert Service (`alert_service_phase3a.py`)

**Multi-channel completion notifications with results display.**

#### Features

- Console detailed results
- Windows MessageBox popup
- PowerShell toast notifications
- Audible alerts (3x beep)
- Email support (optional)
- Results formatting and extraction
- Alert logging

#### Usage

```powershell
# Trigger with detailed results
python alert_service_phase3a.py --detailed

# Quiet mode (minimal output)
python alert_service_phase3a.py --quiet

# Console only (no system notifications)
python alert_service_phase3a.py --no-system

# All alerts enabled (default)
python alert_service_phase3a.py
```

#### Alert Channels

| Channel            | Description              | Enabled                     |
| ------------------ | ------------------------ | --------------------------- |
| Console            | Detailed results table   | ✅ Always                   |
| Windows MessageBox | Popup notification       | 🔧 Optional                 |
| PowerShell Toast   | Windows 10+ notification | 🔧 Optional                 |
| Audible            | 3x system beep           | ✅ Always                   |
| Email              | SMTP notification        | 🔧 Optional (config needed) |

#### Output Example

```
🔔 TRAINING COMPLETION ALERT 🔔

Training completed successfully!
Results available: s:\Ryot\logs_scaled\phase3_stage3a_comparison.json

=================== 📊 RESULTS SUMMARY ===================

BASELINE TRAINING:
  ├─ Total Time: 412.3 seconds
  ├─ Start Loss: 7.523
  ├─ Final Loss: 6.218
  ├─ Throughput: 11.4 tokens/sec
  └─ Status: ✅ COMPLETE

OPTIMIZED TRAINING:
  ├─ Total Time: 289.7 seconds
  ├─ Start Loss: 0.621
  ├─ Final Loss: 0.384
  ├─ Throughput: 11.8 tokens/sec
  └─ Status: ✅ COMPLETE

SPEEDUP ANALYSIS:
  ├─ Baseline Time: 412.3 sec
  ├─ Optimized Time: 289.7 sec
  ├─ Improvement: 1.42x faster
  ├─ Speedup %: 42.3%
  └─ ✅ Target Achieved: YES (≥25% required)

PHASE 1 COMPONENTS VERIFIED:
  ✅ KernelOptimizer (BitNet parallel tuning)
  ✅ SemanticCompressor (MRL + Binary Quantization)
  ✅ InferenceScalingEngine (RLVR Speculative Decoding)

⏰ Timestamp: 2025-03-14 10:25:42 UTC
📋 Full results: phase3_stage3a_comparison.json
🗂️  Checkpoints: checkpoints_scaled/*.pt
📊 Logs: logs_scaled/

NEXT STEPS:
→ Phase 3a validation: ✅ PASSED (speedup ≥25%)
→ Ready for Phase 3b (Production Server)
```

---

## 📈 Expected Training Timeline

Based on Phase 2 results (38.2% speedup over 129.6s → 80.1s):

### Baseline Phase

- **Expected Duration**: ~400 seconds (6.7 minutes)
- **Model**: 9.5M parameters (70x larger than Phase 2)
- **Loss Journey**: 7.5 → 6.2

### Optimized Phase

- **Expected Duration**: ~290 seconds (4.8 minutes) at 37% baseline speedup
- **Model**: Same 9.5M parameters with Phase 1 optimizations
- **Loss Journey**: 0.62 → 0.38

### Total Expected Time

- **Minimum**: ~10 minutes (both phases + overhead)
- **Typical**: ~12-15 minutes
- **Maximum**: ~20 minutes (if slowdowns occur)

### Speedup Target

- **Goal**: ≥25% speedup on larger model
- **Phase 2 Result**: 38.2% speedup
- **Expected Phase 3a**: Similar or better (15-45% range likely)

---

## 🎯 Recommended Workflows

### Workflow 1: Simple Wait (RECOMMENDED FOR QUICK SESSIONS)

```powershell
# Start orchestrator in wait mode, get alerts when done
python orchestrator_phase3a.py --wait

# Estimated: 10-20 minutes, fully automated
# Output: Console alerts + detailed results
```

### Workflow 2: Real-Time Tracking

```powershell
# Terminal 1: Start monitor (detailed real-time)
python monitor_phase3a_training.py

# Terminal 2: Periodic checks (background monitoring)
python status_checker_phase3a.py --periodic 60

# Both feed into alert system automatically
```

### Workflow 3: Unattended Monitoring

```powershell
# Start in background and check later
python status_checker_phase3a.py --periodic 60 > training_checks.log &

# Later, check status anytime
python status_checker_phase3a.py --once

# Or wait for completion
python status_checker_phase3a.py --wait
```

### Workflow 4: Integration with Other Tools

```powershell
# Export status as JSON for scripting
$status = python status_checker_phase3a.py --once | ConvertFrom-Json

# Check if complete
if ($status.has_comparison_json) {
    Write-Host "Training complete!"
    # Trigger other processes...
}

# Or query every minute
while ((python status_checker_phase3a.py --once | ConvertFrom-Json).running) {
    Start-Sleep -Seconds 60
}
Write-Host "Training finished!"
```

---

## 📁 Output Files

### Monitoring Logs

```
logs_scaled/
├─ monitor.log                         # Monitor component output
├─ orchestrator.log                    # Orchestrator master log
├─ alerts.log                          # Alert system events
└─ phase3_stage3a_training.log        # Training script output (if captured)
```

### Status Files

```
./
├─ phase3a_status.json                 # Status snapshot (updated by checker)
└─ phase3_stage3a_comparison.json     # FINAL RESULTS (created at completion)
```

### Checkpoint Files

```
checkpoints_scaled/
├─ scaled_model_epoch_0.pt             # Baseline epoch checkpoints
├─ scaled_model_epoch_1.pt
├─ ... (9 more)
├─ scaled_model_epoch_9.pt             # Final checkpoint
└─ (Plus optimized epoch checkpoints)
```

---

## 🔍 Troubleshooting

### Process Not Found

```
❌ Process not found for train_scaled_model.py

Solution: Make sure training script is still running
  python s:\Ryot\train_scaled_model.py
```

### Results Not Generated

```
⚠️  phase3_stage3a_comparison.json not found

Solution: Training may still be running, or completed without writing results
  - Check: python status_checker_phase3a.py --once
  - Verify: Monitor shows completion detected?
  - Check logs_scaled/ for errors
```

### Memory Warnings

```
⚠️  High memory usage detected (4 GB+)

Normal: Phase 3a model is 9.5M parameters = expected ~2-3 GB
Action: May need to reduce batch size or run on GPU (if available)
```

### Timeout Occurred

```
⚠️  Timeout waiting for training completion

Solution:
  - Check if training is still running: tasklist | grep python
  - Monitor logs: tail -f logs_scaled/monitor.log
  - Manually trigger alerts: python alert_service_phase3a.py
```

---

## 🛠️ Advanced Configuration

### Modify Check Intervals

All tools have configurable intervals:

```powershell
# Monitor: Check every 5 seconds (instead of 10)
# Edit monitor_phase3a_training.py line 25:
check_interval = 5

# Checker: Check every 15 seconds
python status_checker_phase3a.py --periodic 15
```

### Extend Timeout

```powershell
# Wait up to 40 minutes instead of 20
# Edit orchestrator_phase3a.py or modify call:
# Internal: timeout=2400 (seconds)
```

### Custom Alert Channels

Edit `alert_service_phase3a.py`:

- Add email configuration in alert_service_phase3a.py line ~200
- Add Slack webhook integration
- Add custom notification system

---

## 📞 Reference

### Key Files Locations

| File            | Location                                                      |
| --------------- | ------------------------------------------------------------- |
| Training Script | `s:\Ryot\train_scaled_model.py`                               |
| Config          | `s:\Ryot\RYZEN-LLM\configs\scaled_model_training_config.yaml` |
| Monitor Tool    | `s:\Ryot\monitor_phase3a_training.py`                         |
| Checker Tool    | `s:\Ryot\status_checker_phase3a.py`                           |
| Alert Tool      | `s:\Ryot\alert_service_phase3a.py`                            |
| Orchestrator    | `s:\Ryot\orchestrator_phase3a.py`                             |

### Expected Results

| Metric         | Target   | Phase 2 Result | Expected Phase 3a      |
| -------------- | -------- | -------------- | ---------------------- |
| Speedup        | ≥25%     | 38.2%          | 15-45% (likely 30-40%) |
| Model Size     | 8x       | N/A            | 9.5M params ✅         |
| Baseline Time  | N/A      | N/A            | ~400 seconds           |
| Optimized Time | N/A      | N/A            | ~260-290 seconds       |
| Final Loss     | Converge | N/A            | <0.5 expected          |

---

## 🎓 How to Get Help

1. **Check logs**: `cat logs_scaled/monitor.log` or `type logs_scaled\orchestrator.log`
2. **Check status**: `python status_checker_phase3a.py --once --verbose`
3. **Review results**: `cat logs_scaled/phase3_stage3a_comparison.json` (at completion)
4. **Monitor output**: `cat logs_scaled/phase3_stage3a_training.log` (if available)

---

## ✅ Verification Checklist

Before launching:

- [ ] Training script exists: `test-path s:\Ryot\train_scaled_model.py`
- [ ] All monitor scripts exist: `test-path s:\Ryot\{monitor,status_checker,alert_service,orchestrator}_phase3a.py`
- [ ] Python available: `python --version`
- [ ] psutil available: `python -c "import psutil"`
- [ ] Enough disk space in logs_scaled/ and checkpoints_scaled/ (recommend 2 GB)

```powershell
# Quick validation
python orchestrator_phase3a.py --wait
```

---

**Last Updated**: 2025-03-14  
**Phase**: 3a Training Monitoring & Orchestration  
**Status**: Ready for deployment
