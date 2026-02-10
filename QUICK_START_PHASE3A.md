# 🚀 PHASE 3a TRAINING - QUICK START GUIDE

**Status**: Training actively running (Epoch 8/10 - 80% complete)  
**Estimated Completion**: 5-10 minutes  
**Next Action**: Launch monitoring tools (pick your preferred method below)

---

## ⚡ 30-Second Quick Start

### Option 1: Windows - Interactive Menu (EASIEST)

```cmd
launch_orchestrator.bat
```

Select option 1 when menu appears. Done!

### Option 2: Windows PowerShell (MOST FEATURES)

```powershell
.\launch_orchestrator.ps1
```

Select option 1 from menu. Done!

### Option 3: Any OS - Direct Python (SIMPLEST CODE)

```bash
python orchestrator_phase3a.py --full
```

Runs until completion (~20 min). Done!

### Option 4: Linux/Mac/Advanced Users

```bash
python monitor_phase3a_training.py &
python status_checker_phase3a.py --periodic 30
```

---

## 📋 What Will Happen

1. ✅ **Start monitoring** (chosen method launches)
2. ✅ **Progress tracked** (updates every 10-30s)
3. ✅ **Training completes** (5-10 min remaining)
4. ✅ **Auto-alerts triggered** (beep + notification)
5. ✅ **Results displayed** (speedup % shown)
6. ✅ **Phase 3a validated** (≥25% speedup confirmed)

---

## 🎯 Expected Timeline

| Phase              | Duration                  | Status                |
| ------------------ | ------------------------- | --------------------- |
| Baseline Training  | ~400 seconds              | ✅ COMPLETE           |
| Optimized Training | ~280 seconds (Epoch 8/10) | ⏳ 5-10 min remaining |
| **Total Expected** | **~680-690 seconds**      |                       |
| Speedup Expected   | **≥42%**                  | (**Goal: ≥25%**)      |

---

## 🔧 Monitoring Infrastructure (5 Tools)

### 1. **orchestrator_phase3a.py** - Master Control

- **What**: Coordinates all monitoring components
- **Best For**: Fully automated "set and forget"
- **Usage**:
  ```bash
  python orchestrator_phase3a.py --full    # Recommended
  python orchestrator_phase3a.py --wait    # Lightweight
  ```
- **Does**: Starts monitor + checker, waits for completion, triggers alerts
- **Output**: Live updates + auto-alerts on completion

### 2. **monitor_phase3a_training.py** - Real-Time Progress

- **What**: Shows epoch/loss updates in real-time
- **Best For**: Watching progress detailed view
- **Usage**:
  ```bash
  python monitor_phase3a_training.py
  ```
- **Shows**: Every 30 seconds (epoch, loss, throughput)
- **Exits**: Automatically when training complete

### 3. **status_checker_phase3a.py** - Lightweight Polling

- **What**: Minimal resource status checks
- **Best For**: Background monitoring
- **Usage**:
  ```bash
  python status_checker_phase3a.py --periodic 30   # Every 30 seconds
  python status_checker_phase3a.py --once           # Single check
  python status_checker_phase3a.py --wait           # Block until complete
  ```
- **Shows**: Epoch, loss, memory, timestamp
- **Outputs**: Console + JSON file

### 4. **alert_service_phase3a.py** - Completion Alerts

- **What**: Multi-channel notifications
- **Best For**: Getting results summary
- **Usage**:
  ```bash
  python alert_service_phase3a.py --detailed
  ```
- **Triggers**: Console message + Windows notification + beep + optional email
- **Shows**: Full results table (baseline vs optimized)

### 5. **launch_orchestrator.bat / .ps1** - GUI Menu

- **What**: Interactive menu interface
- **Best For**: Non-technical Windows users
- **Usage**: Just double-click or run from PowerShell
- **Menu Options**: 9 different monitoring modes + log viewers

---

## 🎮 Pick Your Preferred Method

### Method A: "Just Tell Me When It's Done" (RECOMMENDED)

**Best for**: Users who want automatic, minimal interaction

```cmd
# Windows CMD
launch_orchestrator.bat
# → Select option 1
# → Wait for auto-alert
# → See results

# OR PowerShell
.\launch_orchestrator.ps1
# → Select option 1
# → Wait for auto-alert
# → See results

# OR Any OS
python orchestrator_phase3a.py --full
# → Auto-handles everything
# → See results when complete
```

**What You'll See**:

- Status updates every 30 seconds
- Epoch/loss displayed
- Auto-alert with results table
- Speedup percentage calculated
- Phase 3a validation result

**Duration**: 10-20 minutes

---

### Method B: "Show Me The Progress"

**Best for**: Users who want to see real-time details

**Terminal 1** - Real-time monitor:

```bash
python monitor_phase3a_training.py
```

**Output**: Every 30s - Epoch | Loss | Throughput

**Terminal 2** (optional) - Resource monitor:

```bash
python status_checker_phase3a.py --periodic 60
```

**Output**: Memory, disk, timestamp snapshots

**When Complete**:

```bash
python alert_service_phase3a.py --detailed
```

**Output**: Full results table

---

### Method C: "I Want Minimal Resource Usage"

**Best for**: Background monitoring, lightweight polling

```bash
# Check every 30 seconds in background
python status_checker_phase3a.py --periodic 30 > training_log.txt &

# Check manually anytime
python status_checker_phase3a.py --once

# When done, get alerts
python alert_service_phase3a.py --detailed
```

---

### Method D: "I Need Full Control"

**Best for**: Advanced users, scripting, CI/CD

```bash
# Individual orchestrator control
python orchestrator_phase3a.py --monitor     # Start monitor only
python orchestrator_phase3a.py --checker     # Start checker only
python orchestrator_phase3a.py --wait        # Wait for completion

# Or use individual tools
python monitor_phase3a_training.py &
python status_checker_phase3a.py --periodic 30 &
# ... do other things ...
python alert_service_phase3a.py --detailed  # When ready for results
```

---

## 📊 Understanding the Output

### Progress Display (From Monitor)

```
[14:32:45] 🔄 MONITORING PHASE 3a TRAINING
[14:33:15] Epoch 8/10 | Loss: 0.38 | Throughput: 11.8 tokens/sec
[14:33:45] Epoch 8/10 | Loss: 0.37 | Throughput: 11.9 tokens/sec
[14:34:15] Epoch 9/10 | Loss: 0.35 | Throughput: 11.8 tokens/sec
[14:34:45] Epoch 9/10 | Loss: 0.34 | Throughput: 12.1 tokens/sec [...]
[14:35:15] ✅ TRAINING COMPLETE
```

### Results Display (From Alert)

```
╔════════════════════════════════════════════════════════════════╗
║                     PHASE 3a RESULTS                           ║
╠════════════════════════════════════════════════════════════════╣
║ BASELINE TRAINING                                              ║
║   • Duration: 412.3 seconds                                   ║
║   • Final Loss: 6.21                                          ║
║   • Throughput: 11.4 tokens/sec                               ║
╠════════════════════════════════════════════════════════════════╣
║ OPTIMIZED TRAINING                                            ║
║   • Duration: 289.7 seconds                                   ║
║   • Final Loss: 0.38                                          ║
║   • Throughput: 11.8 tokens/sec                               ║
╠════════════════════════════════════════════════════════════════╣
║ SPEEDUP ANALYSIS                                              ║
║   • Improvement: 1.42x faster                                 ║
║   • Speedup %: 42.3%                                          ║
║   ✅ TARGET ACHIEVED: YES (≥25% required)                     ║
╚════════════════════════════════════════════════════════════════╝
```

---

## ✅ What to Verify Before Launching

- [ ] Python 3.8+ installed: `python --version`
- [ ] psutil available: `python -c "import psutil"`
- [ ] Training script exists: `train_scaled_model.py`
- [ ] Monitoring scripts exist:
  - [ ] `orchestrator_phase3a.py`
  - [ ] `monitor_phase3a_training.py`
  - [ ] `status_checker_phase3a.py`
  - [ ] `alert_service_phase3a.py`

**Easy Check**: Run either launcher first, it validates automatically ✅

---

## 🔔 Alert Signals When Complete

You'll see/hear:

- ✅ **Console message**: "Training Complete - Results:"
- ✅ **Windows notification**: Popup alert (if available)
- ✅ **Audible alert**: System beep x3
- ✅ **Results table**: Speedup % displayed
- ✅ **JSON file**: `phase3_stage3a_comparison.json` created

---

## 📁 Output Files

**During Training**:

```
logs_scaled/
├─ monitor.log          # Monitor progress log
├─ orchestrator.log     # Orchestrator events
└─ alerts.log           # Alert events
```

**Upon Completion**:

```
./
├─ phase3a_status.json           # Current status snapshot
├─ phase3_stage3a_comparison.json # FINAL RESULTS ⭐
└─ logs_scaled/
   └─ orchestrator.log           # Complete execution log
```

**Checkpoints Saved**:

```
checkpoints_scaled/
├─ scaled_model_epoch_0.pt       # Baseline checkpoints
├─ scaled_model_epoch_1.pt
├─ ... (all 20 total)
└─ scaled_model_epoch_19.pt      # Final checkpoint
```

---

## 🎯 Next Steps After Completion

### If Speedup ≥ 25% ✅ (Expected)

```
✅ Phase 3a VALIDATION PASSED
  → Ready for Phase 3b (Production Server)
  → Proceed with scaling to larger models
```

### If Speedup < 25% ⚠️ (Unlikely)

```
⚠️  Phase 3a VALIDATION INCOMPLETE
  → Debug Phase 1 optimization effectiveness
  → Review optimization parameters
  → Retry with adjusted settings
```

---

## 🆘 Troubleshooting

### "Python not found"

```bash
python --version
# If error: Add Python to PATH or use full path to Python
```

### "ModuleNotFoundError: No module named 'psutil'"

```bash
pip install psutil
```

### "Training script not running"

```bash
# Check if already running
ps aux | grep train_scaled_model.py

# Or restart
python train_scaled_model.py
```

### "Monitor shows no output"

```bash
# Check if training directory correct
ls -la logs_scaled/

# Check training actually running
python status_checker_phase3a.py --once
```

### "Training seems stuck"

```bash
# Check resource usage
python status_checker_phase3a.py --once

# View last log entries
tail -20 logs_scaled/monitor.log

# Manual alert trigger
python alert_service_phase3a.py --detailed
```

---

## 📚 Detailed Documentation

For more information, see:

- **MONITORING_GUIDE.md** - Comprehensive tool documentation
- **orchestrator_phase3a.py** - Source code with comments
- **monitor_phase3a_training.py** - Real-time monitor details
- **status_checker_phase3a.py** - Status checking options
- **alert_service_phase3a.py** - Alert configuration

---

## 🎯 DECISION TIME

**Choose your preferred launch method from above:**

### ✅ RECOMMENDED (Simplest)

```cmd
launch_orchestrator.bat
# Select option 1
```

### ✅ ALSO GOOD (Most Features)

```powershell
.\launch_orchestrator.ps1
# Select option 1
```

### ✅ FAST & CLEAN

```bash
python orchestrator_phase3a.py --full
```

**⏱️ Expected Timeline**:

- Launch tool → Training runs → Completion alert in 10-20 minutes
- Results automatically displayed
- Phase 3a validation complete

---

## 📞 Quick Reference

| Action                     | Command                                   | Time   |
| -------------------------- | ----------------------------------------- | ------ |
| Full auto monitoring       | `orchestrator_phase3a.py --full`          | 20 min |
| Real-time progress         | `monitor_phase3a_training.py`             | 10 min |
| Lightweight polling        | `status_checker_phase3a.py --periodic 30` | 10 min |
| One-time check             | `status_checker_phase3a.py --once`        | 5 sec  |
| Manual alerts              | `alert_service_phase3a.py --detailed`     | 2 sec  |
| Interactive menu (Windows) | `launch_orchestrator.bat`                 | 20 min |
| PowerShell menu            | `.\launch_orchestrator.ps1`               | 20 min |

---

**🚀 Ready? Pick your method from this guide and launch!**

**Questions?** See MONITORING_GUIDE.md for detailed explanations.

**Status**: Training running (Epoch 8/10) → **Launch monitoring now** ✅
