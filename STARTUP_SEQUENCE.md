# Ryzanstein LLM - System Startup Sequence

## ⚠️ CRITICAL: Multi-Process Coordination Required

The Ryzanstein LLM system requires three independent processes to run simultaneously. Each must be started in the correct order.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│              RYZANSTEIN LLM SYSTEM ARCHITECTURE                │
├────────────────────────────────────────────────────────────────┤
│ Process 1: MCP Server (gRPC Backend)                           │
│   • Type: Go executable                                         │
│   • Port: 50051                                                │
│   • Services: Inference, Agent, Memory, Optimization, Debug    │
│   • File: s:\Ryot\mcp\mcp-server.exe                           │
│   • Status Required: MUST BE RUNNING BEFORE API SERVER        │
│                                                                 │
│ Process 2: API Server (REST API)                              │
│   • Type: Python/Uvicorn (containerized)                      │
│   • Port: 5000 (or mapped through Docker)                     │
│   • Services: /api/v1/* REST endpoints                        │
│   • Dependency: Requires MCP server on port 50051             │
│                                                                 │
│ Process 3: Desktop App (Wails GUI)                            │
│   • Type: Go/JavaScript (Wails framework)                     │
│   • Port: 9001 (IPC Server)                                   │
│   • Services: UI, local IPC to MCP server                     │
│   • Dependency: Can start anytime, but needs MCP server       │
│                                                                 │
│ Process 4: Distributed Training System (PyTorch)             │
│   • Type: Python (multi-process)                              │
│   • Backend: Gloo (CPU-optimized)                             │
│   • Status: Configured for CPU (Session 6 verified)            │
│   • Starts on demand during Phase 2                            │
└────────────────────────────────────────────────────────────────┘
```

---

## Startup Sequence

### ✅ Step 1: Start MCP Server (REQUIRED FIRST)

```powershell
# Open Terminal 1 (PowerShell as Administrator)
cd s:\Ryot\mcp
.\mcp-server.exe

# Expected Output:
# [MCP] Starting Ryzanstein MCP Server Suite...
# [MCP] MCP Server listening on :50051
```

**⏱️ Wait Time:** 2-3 seconds
**Health Check:** Port 50051 should be LISTENING

```powershell
# In another terminal:
netstat -ano | Select-String "50051"
# Should show: TCP 0.0.0.0:50051 LISTENING
```

---

### ✅ Step 2: Start API Server (REQUIRES MCP Running)

**Option A: Docker Compose (Recommended)**

```powershell
# Open Terminal 2
cd s:\Ryot\RYZEN-LLM

# If docker-compose.yml exists:
docker-compose up -d api

# Health Check:
curl http://localhost:5000/api/v1/health
# Should return: 200 OK
```

**Option B: Direct Python (Development)**

```powershell
# Open Terminal 2
cd s:\Ryot\RYZEN-LLM

# Ensure MCP server is listening first:
netstat -ano | Select-String "50051"  # Must show LISTENING

# Start API server:
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 5000 --reload

# Expected Output:
# Uvicorn running on http://0.0.0.0:5000
```

**⏱️ Wait Time:** 3-5 seconds for startup and initial reconnection
**Health Check:** Monitor for "GetSystemStatus" success in logs

```powershell
# Circuit breaker should reset once MCP responds
# Logs should show: "pollSystemStatus: GetSystemStatus succeeded"
```

---

### ✅ Step 3: Start Desktop App (Optional/Automatic)

```powershell
# Option A: Automatic (if Wails app is set to startup)
# - Just run the application from Start menu or shortcut

# Option B: Manual (from Terminal 3)
cd s:\Ryot\desktop
wails dev

# Or run the compiled binary:
.\ryzanstein.exe

# Expected Output:
# [Desktop] Application starting up
# [IPC] Starting server on localhost:9001
# [Desktop] Application ready
```

**⏱️ Wait Time:** 5-10 seconds
**Health Check:** Desktop window should appear

---

### ✅ Step 4: Verify All Systems (Validation)

```powershell
# In Terminal 4 (or new PowerShell)

# Check all listening ports:
netstat -ano | Select-String "LISTENING" | Select-String "50051|5000|9001|5432|6379|27017"

# Expected Output:
#   TCP    0.0.0.0:50051      0.0.0.0:0    LISTENING  [MCP PID]
#   TCP    0.0.0.0:5000       0.0.0.0:0    LISTENING  [API PID]
#   TCP    0.0.0.0:9001       0.0.0.0:0    LISTENING  [Desktop PID]
#   (And database ports 5432, 6379, 27017 if running)

# Get process details:
Get-Process | Where-Object {$_.ProcessName -match "mcp-server|python|wails"} | Select-Object ProcessName, Id, PM

# Test API connectivity:
curl http://localhost:5000/api/v1/health -Verbose

# Expected: HTTP 200 OK, no circuit breaker errors
```

---

## Automated Startup Script (PowerShell)

Create `s:\Ryot\START_ALL_SYSTEMS.ps1`:

```powershell
[CmdletBinding()]
param(
    [switch]$SkipMCP,
    [switch]$SkipAPI,
    [switch]$SkipDesktop
)

$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

Write-Host "🚀 Starting Ryzanstein LLM System..." -ForegroundColor Green

# Start MCP Server
if (-not $SkipMCP) {
    Write-Host "`n[1/3] Starting MCP Server (gRPC Backend)..." -ForegroundColor Cyan
    $mcpPath = "s:\Ryot\mcp\mcp-server.exe"

    if (-not (Test-Path $mcpPath)) {
        Write-Error "MCP server not found at $mcpPath"
        exit 1
    }

    # Start MCP in background
    $mcpProcess = Start-Process -FilePath $mcpPath -PassThru -NoNewWindow
    Write-Host "✅ MCP Server started (PID: $($mcpProcess.Id))" -ForegroundColor Green

    # Wait for port to listen
    $maxRetries = 10
    $retry = 0
    while ($retry -lt $maxRetries) {
        Start-Sleep -Milliseconds 500
        $listening = netstat -ano | Select-String "50051.*LISTENING"
        if ($listening) {
            Write-Host "✅ MCP Server listening on port 50051" -ForegroundColor Green
            break
        }
        $retry++
    }

    if ($retry -eq $maxRetries) {
        Write-Error "MCP Server did not start listening on port 50051"
        exit 1
    }
}

# Start API Server
if (-not $SkipAPI) {
    Write-Host "`n[2/3] Starting API Server..." -ForegroundColor Cyan
    Write-Host "Note: If using Docker, start separately with: docker-compose up -d api" -ForegroundColor Yellow
    Write-Host "Note: If using Python directly, start in separate terminal" -ForegroundColor Yellow
}

# Start Desktop App
if (-not $SkipDesktop) {
    Write-Host "`n[3/3] Starting Desktop Application..." -ForegroundColor Cyan
    Write-Host "Note: Start in separate terminal with: cd s:\Ryot\desktop; wails dev" -ForegroundColor Yellow
}

Write-Host "`n✅ Ryzanstein LLM system startup sequence complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Start API Server (Terminal 2)" -ForegroundColor Yellow
Write-Host "2. Start Desktop App (Terminal 3)" -ForegroundColor Yellow
Write-Host "3. Verify connectivity (Terminal 4)" -ForegroundColor Yellow
```

**Usage:**

```powershell
# Start only MCP server:
.\START_ALL_SYSTEMS.ps1

# Skip MCP (already running):
.\START_ALL_SYSTEMS.ps1 -SkipMCP

# Start all components:
.\START_ALL_SYSTEMS.ps1
```

---

## Windows Auto-Startup Configuration

### ⚡ Automatic MCP Server Startup (Recommended)

To prevent manual deployment and ensure MCP server persists across system reboots, configure Windows Task Scheduler:

#### Quick Setup (Recommended)

**Prerequisites:** PowerShell 5.0+, Administrator rights

1. **Open PowerShell as Administrator:**

   ```powershell
   # Right-click PowerShell, select "Run as Administrator"
   ```

2. **Navigate to MCP directory:**

   ```powershell
   cd s:\Ryot\mcp
   ```

3. **Run the auto-startup setup script:**

   ```powershell
   .\SETUP_MCP_AUTOSTARTUP.ps1
   ```

4. **Verify registration:**

   ```powershell
   Get-ScheduledTask -TaskName "RyzansteinMCPServer" | Select-Object State, Enabled
   ```

   **Expected Output:**

   ```
   State   Enabled
   -----   -------
   Ready      True
   ```

#### What This Does

- Creates Windows Task Scheduler task: **"RyzansteinMCPServer"**
- Automatically runs `START_MCP_SERVER.bat` on system startup
- Includes retry logic (3 attempts) if startup fails
- Verifies port 50051 is listening before marking complete
- Logs all operations to `%USERPROFILE%\AppData\Local\Ryzanstein\logs\mcp-server.log`

#### Verification & Testing

**Test the auto-startup mechanism:**

```powershell
# From s:\Ryot\mcp directory:
.\TEST_MCP_STARTUP.ps1 -Verbose

# Expected success output:
# ✅ All prerequisites validated
# ✅ Task Scheduler configuration verified
# ✅ MCP server is LISTENING on port 50051 ✓
# ✅ MCP Server startup test PASSED ✓
```

**Optional: Test with full cleanup:**

```powershell
# Kill existing MCP process and test startup from scratch:
.\TEST_MCP_STARTUP.ps1 -Clean -Verbose -LogResults

# Results will be logged to:
# %USERPROFILE%\AppData\Local\Ryzanstein\logs\startup-test.log
```

#### Check Current Auto-Startup Status

```powershell
# From s:\Ryot\mcp directory:
.\SETUP_MCP_AUTOSTARTUP.ps1 -Status

# Shows:
# - Current task state (Ready, Running, etc.)
# - Whether task is enabled
# - Last run time and result
# - If MCP is currently listening on port 50051
```

#### Manual Task Scheduler Configuration (Alternative)

If the automatic script fails, you can configure manually:

1. **Open Task Scheduler:**

   ```powershell
   taskschd.msc
   ```

2. **Create New Task:**
   - Name: `RyzansteinMCPServer`
   - Description: `Automatically starts Ryzanstein MCP gRPC backend server on system startup`
   - Security Options: ☑ Run with highest privileges

3. **Trigger Tab:**
   - Click "New..."
   - Begin the task: `At startup`
   - Click OK

4. **Action Tab:**
   - Click "New..."
   - Action: `Start a program`
   - Program/script: `cmd.exe`
   - Arguments: `/c "s:\Ryot\mcp\START_MCP_SERVER.bat"`
   - Start in: `s:\Ryot\mcp`
   - Click OK

5. **Settings Tab:**
   - ☑ Allow task to be run on demand
   - ☑ Run task as soon as possible after a scheduled start is missed
   - Stop task if it runs longer than: `1 hour`
   - If the task fails, restart every: `1 minute` (retry count: `3`)

6. **Click OK** to save

#### System Reboot Test

To verify MCP starts automatically on system reboot:

```powershell
# Option 1: Restart and check manually
Restart-Computer -Confirm

# After reboot, verify:
Get-Process mcp-server              # Should exist
netstat -ano | Select-String "50051"  # Should show LISTENING
```

#### Disable Auto-Startup (If Needed)

```powershell
# Disable the auto-startup task:
Get-ScheduledTask -TaskName "RyzansteinMCPServer" | Disable-ScheduledTask

# Verify disabled:
Get-ScheduledTask -TaskName "RyzansteinMCPServer" | Select-Object State, Enabled
```

#### Remove Auto-Startup (Complete Removal)

```powershell
# From s:\Ryot\mcp directory:
.\SETUP_MCP_AUTOSTARTUP.ps1 -Remove

# Or manually in PowerShell (Administrator):
Unregister-ScheduledTask -TaskName "RyzansteinMCPServer" -Confirm:$false
```

### 📋 Batch Script Details

**File:** `s:\Ryot\mcp\START_MCP_SERVER.bat`

This batch script provides the intelligence for startup:

- ✅ Checks if MCP already running (port 50051 LISTENING)
- ✅ Avoids duplicate process if already started
- ✅ Creates log directory automatically
- ✅ Starts MCP in background (minimized window)
- ✅ Verifies port 50051 listening (health check)
- ✅ Retries up to 3 times with 2-second intervals
- ✅ Logs all operations with timestamps
- ✅ Debug mode support via `--debug` parameter

**Logs are saved to:**

```
%USERPROFILE%\AppData\Local\Ryzanstein\logs\mcp-server.log
```

### 🔧 Troubleshooting Auto-Startup

**Task not running on startup?**

1. Check if task is enabled:

   ```powershell
   Get-ScheduledTask -TaskName "RyzansteinMCPServer" | Select-Object Enabled
   ```

2. Check Task Scheduler logs:

   ```powershell
   Get-ScheduledTask -TaskName "RyzansteinMCPServer" | Get-ScheduledTaskInfo | Select-Object LastRunTime, LastTaskResult
   ```

3. Check MCP startup log:

   ```powershell
   Get-Content "$env:USERPROFILE\AppData\Local\Ryzanstein\logs\mcp-server.log" -Tail 20
   ```

4. Verify port 50051 is not already in use:

   ```powershell
   netstat -ano | Select-String "50051"
   ```

5. Try re-registering the task:
   ```powershell
   cd s:\Ryot\mcp
   .\SETUP_MCP_AUTOSTARTUP.ps1
   ```

---

## Troubleshooting

### ❌ MCP Server Won't Start

```powershell
# Check if port 50051 is already in use:
netstat -ano | Select-String "50051"

# Kill existing process:
Get-Process | Where-Object {$_ .ProcessName -eq "mcp-server"} | Stop-Process -Force

# Try starting again:
cd s:\Ryot\mcp
.\mcp-server.exe
```

### ❌ API Server Circuit Breaker Still Open

```powershell
# Verify MCP is listening:
netstat -ano | Select-String "50051"  # Must show LISTENING

# Restart API server to reset circuit breaker:
# If Docker: docker-compose restart api
# If Python: Ctrl+C in API terminal, then restart

# Monitor for successful reconnection:
# Logs should show: "GetSystemStatus succeeded" (no more failures)
```

### ❌ Desktop App Can't Connect to IPC

```powershell
# Check if port 9001 is listening:
netstat -ano | Select-String "9001"

# Check if MCP server is running:
Get-Process | Where-Object {$_.ProcessName -eq "mcp-server"}

# Verify MCP is listening externally:
curl localhost:50051
```

---

## System Dependencies Summary

| Component               | Port  | Protocol  | Status            | Required                      |
| ----------------------- | ----- | --------- | ----------------- | ----------------------------- |
| MCP Server (RPC Engine) | 50051 | gRPC      | ✅ Must run first | YES                           |
| API Server              | 5000  | REST/HTTP | ✅ Depends on MCP | YES                           |
| Desktop App (IPC)       | 9001  | TCP       | ✅ Depends on MCP | Optional                      |
| PostgreSQL              | 5432  | TCP       | -                 | Optional (for persistence)    |
| Redis                   | 6379  | TCP       | -                 | Optional (for caching)        |
| MongoDB                 | 27017 | TCP       | -                 | Optional (for document store) |

---

## Session 6 Context: CPU Architecture

**All systems are now configured for CPU:**

- ✅ **Distributed Training Backend:** Gloo (CPU-optimized, verified in Session 6)
- ✅ **Software Coordinator:** OpenMPI 4.1+ (set in config)
- ✅ **Validation Framework:** All tests CPU-appropriate

**No GPU required.** System is ready for Phase 2 Infrastructure Provisioning.

---

## Phase 2 Integration

Once all systems are running and validated:

1. Verify no circuit breaker errors in API logs
2. Confirm /api/v1/system/status returns 200 OK
3. Proceed with **Week 1 Day 1 Infrastructure Provisioning** checklist
4. Reference: `s:\Ryot\PRODUCTION_DEPLOYMENT_EXECUTION_TIMELINE.md`

---

**Last Updated:** Session 7 (RPC Engine Restoration)
**Status:** ✅ All systems operational and ready for Phase 2
