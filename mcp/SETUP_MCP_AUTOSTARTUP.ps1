#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Sets up Windows Task Scheduler to auto-start the MCP server on system boot

.DESCRIPTION
    This script creates a Windows Task Scheduler task that automatically launches
    the Ryzanstein MCP gRPC backend server on system startup with proper error
    handling and logging.

.PARAMETER TaskName
    The name of the Task Scheduler task (default: "RyzansteinMCPServer")

.PARAMETER Username
    Windows username to run the task as (default: current user)

.PARAMETER Silent
    Skip confirmation prompts

.EXAMPLE
    .\SETUP_MCP_AUTOSTARTUP.ps1
    Creates MCP server auto-startup task with confirmation prompts

    .\SETUP_MCP_AUTOSTARTUP.ps1 -Silent
    Creates MCP server auto-startup task without prompts

.NOTES
    Requires Administrator privileges
#>

param(
    [string]$TaskName = "RyzansteinMCPServer",
    [string]$Username = $env:USERNAME,
    [switch]$Silent,
    [switch]$Remove,
    [switch]$Status
)

$ErrorActionPreference = "Stop"

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartupScript = Join-Path $ScriptDir "START_MCP_SERVER.bat"
$TaskDescription = "Automatically starts Ryzanstein MCP gRPC backend server on system startup"
$TaskTrigger = "Startup"

# Color output functions
function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

# ============================================================================
# Function: Display Setup Information
# ============================================================================
function Show-SetupInfo {
    Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
    Write-Host "  Ryzanstein MCP Server - Auto-Startup Setup" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "`nConfiguration:" -ForegroundColor White
    Write-Host "  Task Name:          $TaskName"
    Write-Host "  Startup Script:     $StartupScript"
    Write-Host "  Run As User:        $Username"
    Write-Host "  Trigger:            On System Startup"
    Write-Host "  Description:        $TaskDescription"
    Write-Host ""
}

# ============================================================================
# Function: Validate Prerequisites
# ============================================================================
function Validate-Prerequisites {
    Write-Info "Validating prerequisites..."
    
    # Check if running as Administrator
    $currentPrincipal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Error-Custom "This script must be run as Administrator"
        Write-Error-Custom "Please re-run: Right-click PowerShell and select 'Run as Administrator'"
        exit 1
    }
    Write-Success "Administrator privileges confirmed"
    
    # Check if startup script exists
    if (-not (Test-Path $StartupScript)) {
        Write-Error-Custom "Startup script not found: $StartupScript"
        Write-Info "Expected location: $StartupScript"
        exit 1
    }
    Write-Success "Startup script found: $StartupScript"
    
    # Check if MCP executable exists
    $MCPExe = Join-Path $ScriptDir "mcp-server.exe"
    if (-not (Test-Path $MCPExe)) {
        Write-Warning "Warning: MCP executable not found at $MCPExe"
        Write-Info "The task will still be created, but may fail if executable is missing"
    }
    else {
        Write-Success "MCP executable found: $MCPExe"
    }
    
    Write-Success "All prerequisites validated`n"
}

# ============================================================================
# Function: Remove Existing Task
# ============================================================================
function Remove-ExistingTask {
    Write-Info "Checking for existing task: $TaskName"
    
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    
    if ($existingTask) {
        Write-Warning "Task already exists: $TaskName"
        
        if (-not $Silent) {
            $confirmation = Read-Host "Remove existing task? (Y/N)"
            if ($confirmation -ne "Y" -and $confirmation -ne "y") {
                Write-Info "Keeping existing task"
                return $false
            }
        }
        
        Write-Info "Removing existing task..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
        Write-Success "Task removed successfully`n"
        return $true
    }
    
    return $true
}

# ============================================================================
# Function: Create Scheduled Task
# ============================================================================
function Create-ScheduledTask {
    Write-Info "Creating new scheduled task..."
    
    # Create task action (what to run)
    $taskAction = New-ScheduledTaskAction `
        -Execute "cmd.exe" `
        -Argument "/c `"$StartupScript`"" `
        -WorkingDirectory $ScriptDir
    
    # Create task trigger (when to run)
    $taskTrigger = New-ScheduledTaskTrigger -AtStartup
    
    # Create task settings
    $taskSettings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries:$false `
        -DontStopIfGoingOnBatteries:$false `
        -StartWhenAvailable:$true `
        -RestartCount:3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -RunOnlyIfNetworkAvailable:$false `
        -MultipleInstances IgnoreNew
    
    # Create principal (who runs the task)
    $taskPrincipal = New-ScheduledTaskPrincipal `
        -UserID "$env:COMPUTERNAME\$Username" `
        -LogonType ServiceAccount `
        -RunLevel Highest
    
    # Register the task
    try {
        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $taskAction `
            -Trigger $taskTrigger `
            -Settings $taskSettings `
            -Principal $taskPrincipal `
            -Description $TaskDescription `
            -Force | Out-Null
        
        Write-Success "Scheduled task created successfully"
    }
    catch {
        Write-Error-Custom "Failed to create scheduled task: $_"
        exit 1
    }
}

# ============================================================================
# Function: Enable and Test Task
# ============================================================================
function Enable-AndTestTask {
    Write-Info "Enabling and verifying task..."
    
    # Enable the task
    try {
        Get-ScheduledTask -TaskName $TaskName | Enable-ScheduledTask | Out-Null
        Write-Success "Task enabled"
    }
    catch {
        Write-Warning "Could not enable task: $_"
    }
    
    # Get task details for verification
    $task = Get-ScheduledTask -TaskName $TaskName
    
    Write-Host "`nTask Details:" -ForegroundColor White
    Write-Host "  Task Name:    $($task.TaskName)"
    Write-Host "  Status:       $($task.State)"
    Write-Host "  Enabled:      $($task.Enabled)"
    Write-Host "  Next Run:     At System Startup"
    
    # Optional: Test run
    if (-not $Silent) {
        $testConfirm = Read-Host "`nTest run the task now? (Y/N)"
        if ($testConfirm -eq "Y" -or $testConfirm -eq "y") {
            Write-Info "Running test execution..."
            Start-ScheduledTask -TaskName $TaskName
            Start-Sleep -Seconds 2
            
            # Check if MCP is now listening
            $listening = netstat -ano | Select-String "50051.*LISTENING"
            if ($listening) {
                Write-Success "MCP server is now listening on port 50051! ✓"
            }
            else {
                Write-Warning "MCP server may not be responding yet (it may recover shortly)"
            }
        }
    }
}

# ============================================================================
# Function: Show Status
# ============================================================================
function Show-TaskStatus {
    Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
    Write-Host "  Task Status" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ""
    
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    
    if (-not $task) {
        Write-Error-Custom "Task not found: $TaskName"
        Write-Info "Run this script without parameters to create the task"
        exit 1
    }
    
    Write-Host "Task Name:           $($task.TaskName)"
    Write-Host "Task Path:           $($task.TaskPath)"
    Write-Host "State:               $($task.State)"
    Write-Host "Enabled:             $($task.Enabled)"
    
    # Get last run info
    $lastRunTime = $task | Get-ScheduledTaskInfo | Select-Object -ExpandProperty LastRunTime
    $lastTaskResult = $task | Get-ScheduledTaskInfo | Select-Object -ExpandProperty LastTaskResult
    
    if ($lastRunTime) {
        Write-Host "Last Run Time:       $lastRunTime"
        Write-Host "Last Run Result:     $lastTaskResult"
    }
    else {
        Write-Host "Last Run Time:       Never"
    }
    
    # Check if MCP is currently listening
    Write-Host "`nCurrent Status:" -ForegroundColor White
    $listening = netstat -ano | Select-String "50051.*LISTENING"
    if ($listening) {
        Write-Success "MCP Server is LISTENING on port 50051"
    }
    else {
        Write-Warning "MCP Server is NOT currently listening (may start on next boot)"
    }
    
    Write-Host "`n"
}

# ============================================================================
# Function: Remove Task
# ============================================================================
function Remove-ManagedTask {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    
    if (-not $task) {
        Write-Warning "Task not found: $TaskName"
        exit 0
    }
    
    Write-Warning "Removing scheduled task: $TaskName"
    
    if (-not $Silent) {
        $confirmation = Read-Host "Are you sure? (Y/N)"
        if ($confirmation -ne "Y" -and $confirmation -ne "y") {
            Write-Info "Cancelled"
            exit 0
        }
    }
    
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Success "Task removed successfully"
}

# ============================================================================
# Main Execution
# ============================================================================

if ($Remove) {
    Remove-ManagedTask
    exit 0
}

if ($Status) {
    Show-TaskStatus
    exit 0
}

# Show setup info
Show-SetupInfo

# Validate prerequisites
Validate-Prerequisites

# Remove existing task if necessary
if (-not (Remove-ExistingTask)) {
    exit 0
}

# Create the scheduled task
Create-ScheduledTask

# Enable and test the task
Enable-AndTestTask

Write-Host "`n" + ("=" * 70) -ForegroundColor Green
Write-Host "✅ MCP Server Auto-Startup Setup Complete!" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "`nThe MCP server will now automatically start when your system boots."
Write-Host "You can verify this by:" -ForegroundColor White
Write-Host "  1. Opening Task Scheduler (search 'Task Scheduler')"
Write-Host "  2. Looking for: $TaskName"
Write-Host "  3. Checking it's enabled and set to run at startup"
Write-Host "`nTo check the current status at any time, run:" -ForegroundColor White
Write-Host "  .\SETUP_MCP_AUTOSTARTUP.ps1 -Status" -ForegroundColor Cyan
Write-Host "`nTo remove this auto-startup task, run:" -ForegroundColor White
Write-Host "  .\SETUP_MCP_AUTOSTARTUP.ps1 -Remove" -ForegroundColor Cyan
Write-Host "`n"
