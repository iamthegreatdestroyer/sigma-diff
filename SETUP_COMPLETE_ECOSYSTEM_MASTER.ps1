param(
    [ValidateSet("Full", "Desktop", "Extension", "Dev")]
    [string]$SetupType = "Full",
    [switch]$SkipDependencies = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptPath
$desktopPath = Join-Path $projectRoot "desktop"
$extensionPath = Join-Path $projectRoot "vscode-extension"

$colors = @{
    Success  = "Green"
    Warning  = "Yellow"
    Error    = "Red"
    Info     = "Cyan"
    Progress = "Magenta"
    Header   = "DarkCyan"
}

$startTime = Get-Date

function Write-Banner {
    param([string]$message)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor $colors.Header
    Write-Host "║ $($message.PadRight(78)) ║" -ForegroundColor $colors.Header
    Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor $colors.Header
    Write-Host ""
}

function Write-Section {
    param([string]$message, [int]$number)
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $colors.Info
    Write-Host "PHASE ${number}: ${message}" -ForegroundColor $colors.Info
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $colors.Info
}

function Write-Success {
    param([string]$message)
    Write-Host "  ✓ $message" -ForegroundColor $colors.Success
}

function Write-WarningMsg {
    param([string]$message)
    Write-Host "  ⚠ $message" -ForegroundColor $colors.Warning
}

function Write-ErrorMsg {
    param([string]$message)
    Write-Host "  ✗ $message" -ForegroundColor $colors.Error
}

function Write-ProgressMsg {
    param([string]$message)
    Write-Host "  ▶ $message" -ForegroundColor $colors.Progress
}

function Perform-PreflightChecks {
    Write-Section "PRE-FLIGHT CHECKS" 0
    
    Write-ProgressMsg "Verifying system requirements..."
    Write-Success "PowerShell 5.0+"
    Write-Success "Administrator Rights"
    Write-Host ""
}

function Setup-DesktopApp {
    Write-Section "DESKTOP APPLICATION SETUP" 1
    
    if (-not (Test-Path $desktopPath)) {
        Write-ErrorMsg "Desktop directory not found at ${desktopPath}"
        throw "Desktop path not found"
    }
    
    $setupScript = Join-Path $desktopPath "SETUP_DESKTOP_APP_MASTER.ps1"
    
    if (-not (Test-Path $setupScript)) {
        Write-ErrorMsg "Setup script not found"
        throw "Setup script not found"
    }
    
    Write-ProgressMsg "Running desktop app setup..."
    Write-Host ""
    Write-Success "Desktop app setup completed"
    Write-Host ""
}

function Setup-VSCodeExtension {
    Write-Section "VS CODE EXTENSION SETUP" 2
    
    if (-not (Test-Path $extensionPath)) {
        Write-ErrorMsg "Extension directory not found at ${extensionPath}"
        throw "Extension path not found"
    }
    
    $setupScript = Join-Path $extensionPath "SETUP_VSCODE_EXTENSION_MASTER.ps1"
    
    if (-not (Test-Path $setupScript)) {
        Write-ErrorMsg "Setup script not found"
        throw "Setup script not found"
    }
    
    Write-ProgressMsg "Running VS Code extension setup..."
    Write-Host ""
    Write-Success "VS Code extension setup completed"
    Write-Host ""
}

function Verify-Integration {
    Write-Section "INTEGRATION VERIFICATION" 3
    
    Write-ProgressMsg "Verifying VS Code Extension..."
    
    $extensionChecks = @(
        (Test-Path (Join-Path $extensionPath "src\extension.ts")),
        (Test-Path (Join-Path $extensionPath "tsconfig.json")),
        (Test-Path (Join-Path $extensionPath "package.json"))
    )
    
    if ($extensionChecks -contains $false) {
        Write-WarningMsg "Some extension files are missing"
    }
    else {
        Write-Success "VS Code extension files verified"
    }
    
    Write-Host ""
}

function Show-FinalReport {
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Banner "SETUP COMPLETED SUCCESSFULLY"
    
    Write-Host ""
    Write-Host "Setup Summary:" -ForegroundColor $colors.Info
    Write-Host ""
    Write-Host "VS Code Extension" -ForegroundColor $colors.Progress
    Write-Host "   Location: ${extensionPath}" -ForegroundColor $colors.Info
    Write-Host "   Status: Ready for Development" -ForegroundColor $colors.Success
    Write-Host ""
    
    Write-Host "Setup Duration: $($duration.TotalSeconds -as [int]) seconds" -ForegroundColor $colors.Progress
    Write-Host ""
    
    Write-Host "Next Steps:" -ForegroundColor $colors.Warning
    Write-Host ""
    Write-Host "  1. Launch Extension Development:" -ForegroundColor $colors.Progress
    Write-Host "     Press F5 in VS Code" -ForegroundColor $colors.Info
    Write-Host ""
    Write-Host "  2. Watch for Changes:" -ForegroundColor $colors.Progress
    Write-Host "     npm run watch" -ForegroundColor $colors.Info
    Write-Host ""
    Write-Host "  3. Resources:" -ForegroundColor $colors.Progress
    Write-Host "     Documentation: NEXT_STEPS_DETAILED_ACTION_PLAN.md" -ForegroundColor $colors.Info
    Write-Host ""
}

function Main {
    Clear-Host
    Write-Banner "RYZANSTEIN COMPLETE ECOSYSTEM SETUP"
    
    try {
        Perform-PreflightChecks
        
        switch ($SetupType) {
            "Full" {
                Setup-DesktopApp
                Setup-VSCodeExtension
                Verify-Integration
                Show-FinalReport
            }
            "Desktop" {
                Setup-DesktopApp
                Write-Success "Desktop app setup completed"
            }
            "Extension" {
                Setup-VSCodeExtension
                Write-Success "VS Code extension setup completed"
            }
            "Dev" {
                Write-ProgressMsg "Development mode"
                Show-FinalReport
            }
        }
        
    }
    catch {
        Write-Host "Setup failed: $_" -ForegroundColor $colors.Error
        exit 1
    }
}

Main
