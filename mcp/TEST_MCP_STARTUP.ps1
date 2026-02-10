#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Tests the MCP server auto-startup mechanism

.DESCRIPTION
    This script validates that the MCP server startup mechanism works correctly by:
    1. Checking the current MCP server status
    2. Optionally stopping the current MCP server
    3. Testing the startup script
    4. Verifying that MCP server starts and listens on port 50051
    5. Logging results for troubleshooting

.PARAMETER Clean
    Kill any existing MCP process before testing

.PARAMETER Timeout
    Seconds to wait for MCP server to start (default: 15)

.PARAMETER Verbose
    Show detailed output messages

.PARAMETER LogResults
    Save test results to file

.EXAMPLE
    .\TEST_MCP_STARTUP.ps1
    Test MCP startup with defaults

    .\TEST_MCP_STARTUP.ps1 -Clean -Verbose
    Kill existing MCP, then test startup with detailed output

    .\TEST_MCP_STARTUP.ps1 -Timeout 30 -LogResults
    Test with 30-second timeout and save results to file

.NOTES
    Requires Administrator privileges
#>

param(
    [switch]$Clean,
    [int]$Timeout = 15,
    [switch]$Verbose,
    [switch]$LogResults,
    [switch]$TaskOnly
)

$ErrorActionPreference = "Stop"
$VerbosePreference = if ($Verbose) { "Continue" } else { "SilentlyContinue" }

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartupScript = Join-Path $ScriptDir "START_MCP_SERVER.bat"
$MCPExe = Join-Path $ScriptDir "mcp-server.exe"
$LogDir = Join-Path $env:USERPROFILE "AppData\Local\Ryzanstein\logs"
$LogFile = Join-Path $LogDir "startup-test.log"
$MCPPort = 50051

# Create log directory if testing with logging
if ($LogResults -and -not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# ============================================================================
# Helper Functions
# ============================================================================

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
    if ($LogResults) { Add-Content $LogFile "[INFO] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Message" }
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
    if ($LogResults) { Add-Content $LogFile "[SUCCESS] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Message" }
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
    if ($LogResults) { Add-Content $LogFile "[WARNING] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Message" }
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
    if ($LogResults) { Add-Content $LogFile "[ERROR] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Message" }
}

function Write-Debug-Info {
    param([string]$Message)
    Write-Verbose $Message
    if ($LogResults) { Add-Content $LogFile "[DEBUG] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Message" }
}

# ============================================================================
# Function: Kill Existing MCP Process
# ============================================================================
function Stop-ExistingMCP {
    Write-Info "Checking for existing MCP server processes..."
    
    $processes = Get-Process -Name "mcp-server" -ErrorAction SilentlyContinue
    
    if ($processes) {
        if ($processes -is [array]) {
            $count = $processes.Count
        }
        else {
            $count = 1
        }
        
        Write-Warning "Found $count MCP process(es)"
        
        if ($Clean) {
            Write-Info "Stopping existing MCP process(es)..."
            Stop-Process -Name "mcp-server" -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
            Write-Success "MCP process(es) stopped"
        }
        else {
            Write-Warning "Use -Clean flag to stop existing processes before testing"
            return $false
        }
    }
    else {
        Write-Success "No existing MCP processes found"
    }
    
    return $true
}

# ============================================================================
# Function: Validate Prerequisites
# ============================================================================
function Validate-Prerequisites {
    Write-Info "Validating prerequisites..."
    
    # Check startup script exists
    if (-not (Test-Path $StartupScript)) {
        Write-Error-Custom "Startup script not found: $StartupScript"
        return $false
    }
    Write-Debug-Info "Startup script found: $StartupScript"
    
    # Check MCP executable exists
    if (-not (Test-Path $MCPExe)) {
        Write-Warning "MCP executable not found at: $MCPExe"
        Write-Warning "The startup script may still attempt to find it in PATH"
    }
    else {
        Write-Debug-Info "MCP executable found: $MCPExe"
    }
    
    # Check netstat available
    if (-not (Get-Command netstat -ErrorAction SilentlyContinue)) {
        Write-Error-Custom "netstat command not found (required for port verification)"
        return $false
    }
    Write-Debug-Info "netstat utility available"
    
    Write-Success "All prerequisites validated"
    return $true
}

# ============================================================================
# Function: Test Task Scheduler
# ============================================================================
function Test-ScheduledTask {
    $TaskName = "RyzansteinMCPServer"
    
    Write-Info "Checking Task Scheduler registration..."
    
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    
    if (-not $task) {
        Write-Warning "Task not found in Task Scheduler: $TaskName"
        Write-Info "Run SETUP_MCP_AUTOSTARTUP.ps1 to register the auto-startup task"
        return $false
    }
    
    Write-Debug-Info "Task found: $($task.TaskName)"
    Write-Debug-Info "Task state: $($task.State)"
    Write-Debug-Info "Task enabled: $($task.Enabled)"
    
    if ($task.State -ne "Ready" -and $task.State -ne "Running") {
        Write-Warning "Task is not in Ready state: $($task.State)"
    }
    
    if (-not $task.Enabled) {
        Write-Warning "Task is disabled"
        Write-Info "Enable with: Get-ScheduledTask -TaskName '$TaskName' | Enable-ScheduledTask"
    }
    
    Write-Success "Task Scheduler configuration verified"
    return $true
}

# ============================================================================
# Function: Run Startup Test
# ============================================================================
function Invoke-StartupTest {
    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "  Running MCP Startup Test" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host ""
    
    Write-Info "Executing startup script: $StartupScript"
    
    # Execute the startup script
    try {
        & cmd.exe /c $StartupScript 2>&1 | ForEach-Object {
            Write-Debug-Info "Script output: $_"
        }
        Write-Debug-Info "Startup script execution completed"
    }
    catch {
        Write-Error-Custom "Failed to execute startup script: $_"
        return $false
    }
    
    return $true
}

# ============================================================================
# Function: Wait for MCP Server
# ============================================================================
function Wait-ForMCPServer {
    Write-Info "Waiting for MCP server to start and listen on port $MCPPort..."
    
    $startTime = Get-Date
    $found = $false
    
    for ($i = 0; $i -lt $Timeout; $i++) {
        # Check if port is listening
        $listening = netstat -ano | Select-String "$MCPPort.*LISTENING"
        
        if ($listening) {
            $found = $true
            Write-Success "MCP server is LISTENING on port $MCPPort ✓"
            Write-Debug-Info "Port verification output: $listening"
            break
        }
        
        $elapsed = (Get-Date) - $startTime
        $remaining = $Timeout - $elapsed.TotalSeconds
        
        Write-Debug-Info "Waiting... (${remaining:F1}s remaining)"
        Start-Sleep -Seconds 1
    }
    
    if (-not $found) {
        Write-Error-Custom "Timeout: MCP server did not start listening within ${Timeout}s"
        
        # Debug: Check if process is running at all
        $process = Get-Process -Name "mcp-server" -ErrorAction SilentlyContinue
        if ($process) {
            Write-Warning "MCP process is running but not listening on port $MCPPort"
            Write-Warning "Process info: $($process | Format-Table -AutoSize | Out-String)"
        }
        else {
            Write-Error-Custom "MCP process is not running"
        }
        
        return $false
    }
    
    return $true
}

# ============================================================================
# Function: Generate Test Report
# ============================================================================
function Generate-TestReport {
    param([bool]$Success)
    
    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "  Test Summary" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host ""
    
    if ($Success) {
        Write-Success "MCP Server startup test PASSED ✓"
        Write-Host "`nMCP Server Status:" -ForegroundColor White
        
        # Get process info
        $process = Get-Process -Name "mcp-server" -ErrorAction SilentlyContinue
        if ($process) {
            if ($process -is [array]) {
                $process = $process[0]
            }
            Write-Host "  Process ID:      $($process.Id)"
            Write-Host "  Memory Usage:    $($process.WorkingSet / 1MB) MB"
            Write-Host "  CPU Usage:       $($process.CPU) seconds"
            Write-Host "  Start Time:      $($process.StartTime)"
        }
        
        # Get port info
        $port = netstat -ano | Select-String "$MCPPort.*LISTENING"
        Write-Host "  Port Status:     LISTENING on port $MCPPort"
        Write-Host "  Netstat Output:  $port"
        
    }
    else {
        Write-Error-Custom "MCP Server startup test FAILED ✗"
        Write-Host "`nTroubleshooting Steps:" -ForegroundColor Yellow
        Write-Host "  1. Check that $MCPExe exists"
        Write-Host "  2. Verify $StartupScript is executable"
        Write-Host "  3. Check logs in: $LogDir"
        Write-Host "  4. Run the startup script manually to see error output"
        Write-Host "  5. Verify no other services are using port $MCPPort"
    }
    
    if ($LogResults) {
        Write-Host "`nResults logged to: $LogFile" -ForegroundColor White
    }
    
    Write-Host ""
}

# ============================================================================
# Main Execution
# ============================================================================

$testResult = $true

Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host "  Ryzanstein MCP Server - Startup Test" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

Write-Info "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Info "Script directory: $ScriptDir"
Write-Info "Timeout: ${Timeout}s"

if ($LogResults) {
    if (Test-Path $LogFile) {
        Remove-Item $LogFile -Force
    }
    Write-Info "Logging enabled: $LogFile"
}

Write-Host ""

# Validate prerequisites
if (-not (Validate-Prerequisites)) {
    $testResult = $false
    Generate-TestReport $false
    exit 1
}

Write-Host ""

# Check existing processes
if (-not (Stop-ExistingMCP)) {
    if (-not $Clean) {
        Write-Info "Proceeding with existing MCP process (may affect test results)"
    }
}

Write-Host ""

# Test Task Scheduler registration
if ($TaskOnly -or $true) {
    if (-not (Test-ScheduledTask)) {
        $testResult = $false
    }
    Write-Host ""
}

# Run startup test
if (Invoke-StartupTest) {
    # Wait for server to start
    if (-not (Wait-ForMCPServer)) {
        $testResult = $false
    }
}
else {
    $testResult = $false
}

Write-Host ""

# Generate report
Generate-TestReport $testResult

Write-Info "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Return appropriate exit code
if ($testResult) {
    exit 0
}
else {
    exit 1
}
