#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Phase 3a Training Orchestrator - PowerShell Launcher
    Interactive menu for launching monitoring tools

.DESCRIPTION
    Provides easy access to monitoring, status checking, and alert services
    for Phase 3a training execution.

.EXAMPLE
    .\launch_orchestrator.ps1
    # Interactive menu-driven interface

.NOTES
    Requires Python 3.8+ with psutil installed
    Run from s:\Ryot directory
#>

param(
    [ValidateSet('full', 'wait', 'monitor', 'checker', 'alert', 'status', 'logs', 'debug')]
    [string]$Mode = $null,
    
    [int]$CheckInterval = 30,
    
    [switch]$Verbose = $false,
    
    [switch]$NoMenu = $false
)

# Configuration
$Script:ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Script:ScriptRoot

function Test-PythonAvailable {
    try {
        $version = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Python $version"
            return $true
        }
    }
    catch {}
    return $false
}

function Test-Environment {
    Write-Host "`n🔍 CHECKING ENVIRONMENT" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    
    # Check Python
    if (-not (Test-PythonAvailable)) {
        Write-Host "❌ Python not found! Please install Python 3.8+ or add to PATH." -ForegroundColor Red
        return $false
    }
    
    # Check psutil
    Write-Host -NoNewline "Checking psutil... "
    $result = python -c "import psutil" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅" -ForegroundColor Green
    }
    else {
        Write-Host "❌ (installing...)" -ForegroundColor Yellow
        python -m pip install psutil -q
    }
    
    # Check scripts exist
    Write-Host "Checking monitoring scripts... " -NoNewline
    $scripts = @(
        "orchestrator_phase3a.py",
        "monitor_phase3a_training.py",
        "status_checker_phase3a.py",
        "alert_service_phase3a.py"
    )
    
    $missing = @()
    foreach ($script in $scripts) {
        if (-not (Test-Path $script)) {
            $missing += $script
        }
    }
    
    if ($missing.Count -eq 0) {
        Write-Host "✅ All scripts found" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Missing: $($missing -join ', ')" -ForegroundColor Red
        return $false
    }
    
    # Check training script
    Write-Host -NoNewline "Checking training script... "
    if (Test-Path "train_scaled_model.py") {
        Write-Host "✅" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  Training script not found" -ForegroundColor Yellow
    }
    
    Write-Host "✅ Environment check complete`n" -ForegroundColor Green
    return $true
}

function Invoke-OrchestratorFullMode {
    Write-Host "`n🚀 STARTING FULL ORCHESTRATION" -ForegroundColor Cyan
    Write-Host "This will run for up to 20 minutes" -ForegroundColor Yellow
    Write-Host "Starting monitor, checker, and awaiting completion...`n" -ForegroundColor Gray
    
    python orchestrator_phase3a.py --full
}

function Invoke-OrchestratorWaitMode {
    Write-Host "`n⏳ STARTING WAIT MODE" -ForegroundColor Cyan
    Write-Host "Monitoring background training for completion`n" -ForegroundColor Gray
    
    python orchestrator_phase3a.py --wait
}

function Invoke-Monitor {
    Write-Host "`n📊 STARTING PROGRESS MONITOR" -ForegroundColor Cyan
    Write-Host "Real-time progress tracking (auto-exits on completion)`n" -ForegroundColor Gray
    
    python monitor_phase3a_training.py
}

function Invoke-StatusChecker {
    Write-Host "`n📈 STARTING STATUS CHECKER" -ForegroundColor Cyan
    Write-Host "Periodic status updates every ${CheckInterval}s`n" -ForegroundColor Gray
    
    python status_checker_phase3a.py --periodic $CheckInterval $(if ($Verbose) { '--verbose' })
}

function Invoke-StatusOnce {
    Write-Host "`n📍 STATUS CHECK (ONCE)" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    
    python status_checker_phase3a.py --once $(if ($Verbose) { '--verbose' })
}

function Invoke-AlertService {
    Write-Host "`n🔔 TRIGGERING ALERT SERVICE" -ForegroundColor Cyan
    Write-Host "Sending completion notifications...`n" -ForegroundColor Gray
    
    python alert_service_phase3a.py --detailed
}

function Show-TailLog {
    param([string]$LogFile, [int]$Lines = 20)
    
    if (Test-Path $LogFile) {
        Write-Host "📋 $LogFile (last $Lines lines):" -ForegroundColor Cyan
        Write-Host ("=" * 80) -ForegroundColor Cyan
        
        Get-Content $LogFile -Tail $Lines
        
        Write-Host ("=" * 80) -ForegroundColor Cyan
    }
    else {
        Write-Host "⚠️  Log file not found: $LogFile" -ForegroundColor Yellow
    }
}

function Show-DebugInfo {
    Write-Host "`n🔍 DEBUG INFORMATION" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    
    Write-Host "`n📍 Working Directory:"
    Get-Location | Select-Object -ExpandProperty Path
    
    Write-Host "`n📂 Directory Contents:"
    ls -Filter "*phase3a*" -File | Select-Object Name, Length, LastWriteTime
    
    Write-Host "`n📊 Status File:"
    if (Test-Path "phase3a_status.json") {
        Get-Content phase3a_status.json | ConvertFrom-Json | Format-List
    }
    else {
        Write-Host "  (not found)"
    }
    
    Write-Host "`n🔄 Running Processes:"
    Get-Process | Where-Object { $_.Name -like "*python*" } | Select-Object Name, ID, WorkingSet
    
    Write-Host "`n📊 Disk Space:"
    Get-Volume | Select-Object DriveLetter, Size, SizeRemaining | Format-Table -AutoSize
}

function Show-Menu {
    $quick_status = ""
    if (Test-Path "train_scaled_model.py") {
        $proc = Get-Process -Name python -ErrorAction SilentlyContinue | 
        Where-Object { $_.CommandLine -like "*train_scaled_model*" }
        if ($proc) {
            $quick_status = " [TRAINING IN PROGRESS]"
        }
    }
    
    Clear-Host
    Write-Host "`n"
    Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                   🚀 PHASE 3a TRAINING ORCHESTRATOR                           ║" -ForegroundColor Cyan
    Write-Host "║                         PowerShell Launcher                                   ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    
    Write-Host "`n📍 Current Directory: $(Get-Location)" -ForegroundColor Gray
    if ($quick_status) {
        Write-Host "🔴 $($quick_status.Trim())" -ForegroundColor Yellow
    }
    
    Write-Host "`n" 
    Write-Host "SELECT MONITORING MODE:" -ForegroundColor White
    Write-Host "`n  [1] Full Orchestration (RECOMMENDED)" -ForegroundColor Green
    Write-Host "      • Starts monitor and checker"
    Write-Host "      • Waits up to 20 minutes for completion"
    Write-Host "      • Auto-displays results and alerts"
    
    Write-Host "`n  [2] Wait Mode" -ForegroundColor Yellow
    Write-Host "      • Monitor in background"
    Write-Host "      • Blocks until complete"
    
    Write-Host "`n  [3] Monitor Only"
    Write-Host "      • Real-time progress tracking"
    Write-Host "      • Shows epoch/loss every 30 seconds"
    
    Write-Host "`n  [4] Status Checker - Periodic"
    Write-Host "      • Lightweight status snapshots"
    Write-Host "      • Updates every 30 seconds"
    
    Write-Host "`n  [5] Status Checker - Once"
    Write-Host "      • Single status check"
    
    Write-Host "`n  [6] Trigger Alerts (Manual)"
    Write-Host "      • Manually trigger completion alerts"
    
    Write-Host "`n  [7] View Latest Logs"
    Write-Host "      • Display monitor/orchestrator logs"
    
    Write-Host "`n  [8] Debug Information"
    Write-Host "      • Show environment and status details"
    
    Write-Host "`n  [9] Environment Check"
    Write-Host "      • Verify all dependencies"
    
    Write-Host "`n  [Q] Quit"
    
    Write-Host "`n" + ("=" * 80) -ForegroundColor Gray
}

function Start-InteractiveLoop {
    do {
        Show-Menu
        $choice = Read-Host "`nEnter selection"
        
        Write-Host ""
        
        switch ($choice) {
            '1' { Invoke-OrchestratorFullMode }
            '2' { Invoke-OrchestratorWaitMode }
            '3' { Invoke-Monitor }
            '4' { Invoke-StatusChecker }
            '5' { Invoke-StatusOnce }
            '6' { Invoke-AlertService }
            '7' {
                Show-TailLog "logs_scaled\monitor.log" 20
                Write-Host ""
                Show-TailLog "logs_scaled\orchestrator.log" 20
            }
            '8' { Show-DebugInfo }
            '9' {
                if (Test-Environment) {
                    Write-Host "`n✅ All checks passed! Ready to launch monitoring." -ForegroundColor Green
                }
                else {
                    Write-Host "`n❌ Some checks failed. Please address issues above." -ForegroundColor Red
                }
            }
            'Q' { 
                Write-Host "Goodbye!`n" -ForegroundColor Yellow
                exit 0 
            }
            default {
                Write-Host "❌ Invalid selection. Please try again." -ForegroundColor Red
                Start-Sleep -Seconds 2
            }
        }
        
        Write-Host ""
        Read-Host "Press Enter to continue"
        
    } while ($true)
}

function Main {
    # Environment check
    if (-not (Test-Environment)) {
        Write-Error "Environment check failed!"
        exit 1
    }
    
    # Direct mode if specified
    if ($Mode -and -not $NoMenu) {
        switch ($Mode) {
            'full' { Invoke-OrchestratorFullMode }
            'wait' { Invoke-OrchestratorWaitMode }
            'monitor' { Invoke-Monitor }
            'checker' { Invoke-StatusChecker }
            'status' { Invoke-StatusOnce }
            'alert' { Invoke-AlertService }
            'logs' { 
                Show-TailLog "logs_scaled\monitor.log" 30
                Show-TailLog "logs_scaled\orchestrator.log" 30
            }
            'debug' { Show-DebugInfo }
        }
    }
    else {
        # Interactive menu
        Start-InteractiveLoop
    }
}

# Run main
Main
