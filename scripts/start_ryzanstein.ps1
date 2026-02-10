# =============================================================================
# RYZANSTEIN UNIFIED LAUNCHER
# Starts all components of the full custom LLM stack
# =============================================================================
#
# Usage:
#   .\start_ryzanstein.ps1              # Start all services
#   .\start_ryzanstein.ps1 -ApiOnly     # Start only Python API
#   .\start_ryzanstein.ps1 -NoDesktop   # Start services without desktop app
#   .\start_ryzanstein.ps1 -Status      # Check service status
#   .\start_ryzanstein.ps1 -Stop        # Stop all services
#
# =============================================================================

param(
    [switch]$ApiOnly,         # Start only Python API
    [switch]$NoDesktop,       # Start services without desktop
    [switch]$Status,          # Check service status
    [switch]$Stop,            # Stop all services
    [switch]$Verbose,         # Verbose output
    [switch]$Help             # Show help
)

$ErrorActionPreference = "Continue"
$ProjectRoot = "S:\Ryot"
$RyzenLLMRoot = "$ProjectRoot\RYZEN-LLM"

# Service ports
$PYTHON_API_PORT = 8000
$MCP_SERVER_PORT = 50051
$DESKTOP_PORT = 34115

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function Write-Status {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

function Wait-ForPort {
    param([int]$Port, [int]$TimeoutSeconds = 30, [string]$ServiceName = "Service")
    
    $startTime = Get-Date
    while (-not (Test-PortInUse -Port $Port)) {
        if ((Get-Date) - $startTime -gt (New-TimeSpan -Seconds $TimeoutSeconds)) {
            Write-Status "  ⚠️ $ServiceName did not start within $TimeoutSeconds seconds" "Yellow"
            return $false
        }
        Start-Sleep -Milliseconds 500
    }
    return $true
}

function Get-ServiceStatus {
    $status = @{}
    
    # Python API
    $status["Python API (Port $PYTHON_API_PORT)"] = Test-PortInUse -Port $PYTHON_API_PORT
    
    # MCP Server
    $status["MCP Server (Port $MCP_SERVER_PORT)"] = Test-PortInUse -Port $MCP_SERVER_PORT
    
    # Desktop
    $desktopProc = Get-Process -Name "ryzanstein" -ErrorAction SilentlyContinue
    $status["Desktop App"] = $null -ne $desktopProc
    
    return $status
}

# =============================================================================
# BANNER
# =============================================================================

Write-Host @"

╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   ██████╗ ██╗   ██╗███████╗ █████╗ ███╗   ██╗███████╗████████╗███████╗██╗███╗   ║
║   ██╔══██╗╚██╗ ██╔╝╚══███╔╝██╔══██╗████╗  ██║██╔════╝╚══██╔══╝██╔════╝██║████╗  ║
║   ██████╔╝ ╚████╔╝   ███╔╝ ███████║██╔██╗ ██║███████╗   ██║   █████╗  ██║██╔██╗ ║
║   ██╔══██╗  ╚██╔╝   ███╔╝  ██╔══██║██║╚██╗██║╚════██║   ██║   ██╔══╝  ██║██║╚██╗║
║   ██║  ██║   ██║   ███████╗██║  ██║██║ ╚████║███████║   ██║   ███████╗██║██║ ╚██║
║   ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚══════╝╚═╝╚═╝  ╚═╝
║                                                                                  ║
║                         UNIFIED LAUNCHER - Your Custom LLM                       ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# =============================================================================
# HELP
# =============================================================================

if ($Help) {
    Write-Host @"
Usage: .\start_ryzanstein.ps1 [options]

Options:
  -ApiOnly      Start only the Python API server
  -NoDesktop    Start all services except the desktop app
  -Status       Show status of all services
  -Stop         Stop all running services
  -Verbose      Show verbose output
  -Help         Show this help message

Services:
  1. Python API    - FastAPI server on port $PYTHON_API_PORT
  2. MCP Server    - gRPC server on port $MCP_SERVER_PORT  
  3. Desktop App   - Wails GUI (connects to MCP)

URLs:
  API Docs:     http://localhost:$PYTHON_API_PORT/docs
  OpenAI API:   http://localhost:$PYTHON_API_PORT/v1/chat/completions

"@
    exit 0
}

# =============================================================================
# STATUS CHECK
# =============================================================================

if ($Status) {
    Write-Status "Service Status:" "Yellow"
    
    $status = Get-ServiceStatus
    
    foreach ($item in $status.GetEnumerator()) {
        if ($item.Value) {
            Write-Status "  ✅ $($item.Key) - Running" "Green"
        }
        else {
            Write-Status "  ⚪ $($item.Key) - Stopped" "Gray"
        }
    }
    
    exit 0
}

# =============================================================================
# STOP ALL SERVICES
# =============================================================================

if ($Stop) {
    Write-Status "Stopping all services..." "Yellow"
    
    # Stop Python API
    $pythonProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -match "uvicorn|fastapi|server"
    }
    if ($pythonProcs) {
        $pythonProcs | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Status "  ✅ Python API stopped" "Green"
    }
    
    # Stop MCP Server
    $mcpProcs = Get-Process -Name "mcp-server" -ErrorAction SilentlyContinue
    if ($mcpProcs) {
        $mcpProcs | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Status "  ✅ MCP Server stopped" "Green"
    }
    
    # Stop Desktop
    $desktopProcs = Get-Process -Name "ryzanstein" -ErrorAction SilentlyContinue
    if ($desktopProcs) {
        $desktopProcs | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Status "  ✅ Desktop App stopped" "Green"
    }
    
    Write-Status "All services stopped" "Green"
    exit 0
}

# =============================================================================
# START SERVICES
# =============================================================================

Write-Status "Starting Ryzanstein services..." "Yellow"

# Set up environment
$env:PYTHONPATH = "$RyzenLLMRoot\src;$RyzenLLMRoot\build\python\ryzanstein_llm"
$env:RYZEN_MODEL_PATH = "$RyzenLLMRoot\models\bitnet\3b"

# -----------------------------------------------------------------------------
# 1. Python API Server
# -----------------------------------------------------------------------------

Write-Status "[1/3] Starting Python API Server..." "Cyan"

$apiRunning = Test-PortInUse -Port $PYTHON_API_PORT

if ($apiRunning) {
    Write-Status "  ⚠️ Python API already running on port $PYTHON_API_PORT" "Yellow"
}
else {
    $apiScript = @"
import uvicorn
import sys
sys.path.insert(0, r'$RyzenLLMRoot\src')
sys.path.insert(0, r'$RyzenLLMRoot\build\python\ryzanstein_llm')
from api.server import app
uvicorn.run(app, host='127.0.0.1', port=$PYTHON_API_PORT, log_level='info')
"@
    
    # Check if server.py exists, if not create a basic one
    $serverPath = "$RyzenLLMRoot\src\api\server.py"
    if (-not (Test-Path $serverPath)) {
        Write-Status "  Creating API server file..." "Gray"
        New-Item -ItemType Directory -Path "$RyzenLLMRoot\src\api" -Force | Out-Null
        # Will be created by create_api_wrapper step
    }
    
    # Try to start the API
    $apiScriptPath = "$env:TEMP\start_ryzanstein_api.py"
    $apiScript | Out-File -FilePath $apiScriptPath -Encoding UTF8 -Force
    
    try {
        Start-Process -FilePath "python" -ArgumentList $apiScriptPath `
            -WindowStyle Minimized -PassThru -ErrorAction SilentlyContinue | Out-Null
        
        if (Wait-ForPort -Port $PYTHON_API_PORT -TimeoutSeconds 15 -ServiceName "Python API") {
            Write-Status "  ✅ Python API started on http://localhost:$PYTHON_API_PORT" "Green"
        }
    }
    catch {
        Write-Status "  ⚠️ Could not start Python API: $_" "Yellow"
        Write-Status "  Tip: Run manually: python -m uvicorn api.server:app --port $PYTHON_API_PORT" "Gray"
    }
}

if ($ApiOnly) {
    Write-Status "`nAPI-only mode: Stopping here." "Gray"
    Write-Status "API Docs: http://localhost:$PYTHON_API_PORT/docs" "Cyan"
    exit 0
}

# -----------------------------------------------------------------------------
# 2. MCP Server
# -----------------------------------------------------------------------------

Write-Status "[2/3] Starting MCP Server..." "Cyan"

$mcpRunning = Test-PortInUse -Port $MCP_SERVER_PORT

if ($mcpRunning) {
    Write-Status "  ⚠️ MCP Server already running on port $MCP_SERVER_PORT" "Yellow"
}
else {
    $mcpExe = "$ProjectRoot\mcp\mcp-server.exe"
    
    if (Test-Path $mcpExe) {
        Push-Location "$ProjectRoot\mcp"
        try {
            Start-Process -FilePath $mcpExe -WindowStyle Minimized -PassThru | Out-Null
            
            if (Wait-ForPort -Port $MCP_SERVER_PORT -TimeoutSeconds 10 -ServiceName "MCP Server") {
                Write-Status "  ✅ MCP Server started on port $MCP_SERVER_PORT" "Green"
            }
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Status "  ⚠️ MCP Server executable not found at $mcpExe" "Yellow"
        Write-Status "  Tip: Run .\scripts\build_complete_stack.ps1 first" "Gray"
    }
}

if ($NoDesktop) {
    Write-Status "`nNo-desktop mode: Skipping desktop app." "Gray"
    Write-Status "Services running - use -Status to check" "Cyan"
    exit 0
}

# -----------------------------------------------------------------------------
# 3. Desktop App
# -----------------------------------------------------------------------------

Write-Status "[3/3] Starting Desktop App..." "Cyan"

$desktopProc = Get-Process -Name "ryzanstein" -ErrorAction SilentlyContinue

if ($desktopProc) {
    Write-Status "  ⚠️ Desktop app already running" "Yellow"
}
else {
    $desktopExe = "$ProjectRoot\desktop\build\bin\ryzanstein.exe"
    
    if (Test-Path $desktopExe) {
        Start-Process -FilePath $desktopExe -PassThru | Out-Null
        Start-Sleep -Seconds 2
        
        $desktopProc = Get-Process -Name "ryzanstein" -ErrorAction SilentlyContinue
        if ($desktopProc) {
            Write-Status "  ✅ Desktop App started" "Green"
        }
    }
    else {
        Write-Status "  ⚠️ Desktop executable not found at $desktopExe" "Yellow"
        Write-Status "  Tip: Run .\scripts\build_complete_stack.ps1 first" "Gray"
    }
}

# =============================================================================
# SUMMARY
# =============================================================================

Write-Host @"

╔══════════════════════════════════════════════════════════════════════════════════╗
║                          RYZANSTEIN IS RUNNING                                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
"@ -ForegroundColor Green

$finalStatus = Get-ServiceStatus
foreach ($item in $finalStatus.GetEnumerator()) {
    $statusIcon = if ($item.Value) { "✅" } else { "⚪" }
    Write-Host "║    $statusIcon $($item.Key)"
}

Write-Host @"
║
║  Endpoints:
║    📚 API Docs:    http://localhost:$PYTHON_API_PORT/docs
║    🤖 OpenAI API:  http://localhost:$PYTHON_API_PORT/v1/chat/completions
║    🔌 MCP gRPC:    localhost:$MCP_SERVER_PORT
║
║  Commands:
║    Status:  .\start_ryzanstein.ps1 -Status
║    Stop:    .\start_ryzanstein.ps1 -Stop
║
╚══════════════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Status "Your custom LLM is ready. Enjoy! 🚀" "Cyan"
