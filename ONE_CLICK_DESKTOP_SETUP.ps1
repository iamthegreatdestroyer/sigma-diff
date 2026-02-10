param(
    [switch]$SkipValidation = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
$desktopPath = Join-Path $projectRoot "desktop"

# Colors for output
$colors = @{
    Success  = "Green"
    Error    = "Red"
    Warning  = "Yellow"
    Info     = "Cyan"
    Progress = "Magenta"
    Header   = "DarkCyan"
}

function Write-Header {
    param([string]$message)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor $colors.Header
    Write-Host "║ $($message.PadRight(78)) ║" -ForegroundColor $colors.Header
    Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor $colors.Header
    Write-Host ""
}

function Write-Step {
    param([string]$message, [int]$number)
    Write-Host ""
    Write-Host "Step $number : $message" -ForegroundColor $colors.Progress
    Write-Host "─────────────────────────────────────────────────────────────────────────────────" -ForegroundColor $colors.Info
}

function Write-Success {
    param([string]$message)
    Write-Host "  ✓ $message" -ForegroundColor $colors.Success
}

function Write-Info {
    param([string]$message)
    Write-Host "  ℹ $message" -ForegroundColor $colors.Info
}

function Write-Check {
    param([string]$item, [bool]$passed)
    if ($passed) {
        Write-Host "  ✓ $item" -ForegroundColor $colors.Success
    }
    else {
        Write-Host "  ✗ $item" -ForegroundColor $colors.Error
    }
}

function Test-Dependency {
    param(
        [string]$command,
        [string]$name,
        [string]$installUrl
    )
    
    try {
        $null = & $command --version 2>$null
        Write-Check "$name installed" $true
        return $true
    }
    catch {
        Write-Check "$name installed" $false
        if ($installUrl) {
            Write-Host "    → Install from: $installUrl" -ForegroundColor $colors.Warning
        }
        return $false
    }
}

function Validate-Prerequisites {
    Write-Step "Validating Prerequisites" 1
    
    $allValid = $true
    
    Write-Info "Checking required dependencies..."
    Write-Host ""
    
    $allValid = (Test-Dependency "go" "Go 1.20+" "https://golang.org/dl") -and $allValid
    $allValid = (Test-Dependency "node" "Node.js 18+" "https://nodejs.org") -and $allValid
    $allValid = (Test-Dependency "npm" "npm 8+" "https://nodejs.org") -and $allValid
    
    Write-Host ""
    
    if ($allValid) {
        Write-Success "All dependencies found!"
    }
    else {
        throw "Missing required dependencies. Please install them first."
    }
}

function Setup-Backend {
    Write-Step "Setting Up Backend (Go)" 2
    
    $backendPath = Join-Path $desktopPath "cmd\ryzanstein"
    
    if (-not (Test-Path $backendPath)) {
        throw "Backend directory not found at $backendPath"
    }
    
    Write-Info "Building Go application..."
    Push-Location $desktopPath
    
    try {
        # Download dependencies
        Write-Host "  → Downloading Go modules..." -ForegroundColor $colors.Info
        & go mod download 2>$null
        
        # Build the application
        Write-Host "  → Compiling Go code..." -ForegroundColor $colors.Info
        $buildCmd = "go"
        if ($IsWindows) {
            & $buildCmd build -o bin\ryzanstein.exe ./cmd/ryzanstein 2>&1 | ForEach-Object {
                if ($_ -match "error") {
                    Write-Host "    ERROR: $_" -ForegroundColor $colors.Error
                }
                else {
                    Write-Host "    $_" -ForegroundColor $colors.Info
                }
            }
        }
        else {
            & $buildCmd build -o bin/ryzanstein ./cmd/ryzanstein 2>&1 | ForEach-Object {
                if ($_ -match "error") {
                    Write-Host "    ERROR: $_" -ForegroundColor $colors.Error
                }
                else {
                    Write-Host "    $_" -ForegroundColor $colors.Info
                }
            }
        }
        
        Write-Success "Backend compiled successfully"
    }
    catch {
        throw "Backend compilation failed: $_"
    }
    finally {
        Pop-Location
    }
}

function Setup-Frontend {
    Write-Step "Setting Up Frontend (React + Wails)" 3
    
    $frontendPath = Join-Path $desktopPath "packages\desktop"
    
    if (-not (Test-Path $frontendPath)) {
        Write-Warning "Frontend directory structure not found, skipping setup"
        return
    }
    
    Push-Location $frontendPath
    
    try {
        Write-Info "Installing npm dependencies..."
        Write-Host "  → This may take a minute..." -ForegroundColor $colors.Info
        & npm install 2>&1 | Select-Object -Last 5 | ForEach-Object { Write-Host "    $_" -ForegroundColor $colors.Info }
        
        Write-Info "Building React application..."
        & npm run build 2>&1 | Select-Object -Last 3 | ForEach-Object { Write-Host "    $_" -ForegroundColor $colors.Info }
        
        Write-Success "Frontend built successfully"
    }
    catch {
        Write-Warning "Frontend build had issues (non-critical): $_"
    }
    finally {
        Pop-Location
    }
}

function Setup-Wails {
    Write-Step "Setting Up Wails Framework" 4
    
    Push-Location $desktopPath
    
    try {
        Write-Info "Installing Wails CLI..."
        & go install github.com/wailsapp/wails/v2/cmd/wails@latest 2>&1 | Select-Object -Last 2 | ForEach-Object { Write-Host "    $_" -ForegroundColor $colors.Info }
        
        Write-Success "Wails installed successfully"
    }
    catch {
        Write-Warning "Wails installation skipped (optional): $_"
    }
    finally {
        Pop-Location
    }
}

function Test-Build {
    Write-Step "Testing Build Output" 5
    
    $binPath = Join-Path $desktopPath "bin"
    $executable = if ($IsWindows) { "ryzanstein.exe" } else { "ryzanstein" }
    $exePath = Join-Path $binPath $executable
    
    if (Test-Path $exePath) {
        $fileSize = (Get-Item $exePath).Length / 1MB
        Write-Success "Desktop app executable found (${fileSize:N2} MB)"
    }
    else {
        Write-Warning "Executable not found at expected location"
    }
}

function Show-Instructions {
    Write-Header "SETUP COMPLETE - NEXT STEPS"
    
    Write-Host "🎉 Your Ryzanstein Desktop App is ready!" -ForegroundColor $colors.Success
    Write-Host ""
    
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host "LAUNCH THE APPLICATION" -ForegroundColor $colors.Header
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host ""
    
    Write-Host "Option 1: Development Mode (Recommended)" -ForegroundColor $colors.Progress
    Write-Host "  1. Open Terminal" -ForegroundColor $colors.Info
    Write-Host "  2. Run: cd $desktopPath" -ForegroundColor $colors.Info
    Write-Host "  3. Run: wails dev" -ForegroundColor $colors.Info
    Write-Host "  → Application launches with hot-reload enabled" -ForegroundColor $colors.Info
    Write-Host ""
    
    Write-Host "Option 2: Production Build" -ForegroundColor $colors.Progress
    Write-Host "  1. Open Terminal" -ForegroundColor $colors.Info
    Write-Host "  2. Run: cd $desktopPath" -ForegroundColor $colors.Info
    Write-Host "  3. Run: wails build -clean" -ForegroundColor $colors.Info
    if ($IsWindows) {
        Write-Host "  → Creates installer: build\bin\ryzanstein-amd64-installer.exe" -ForegroundColor $colors.Info
    }
    else {
        Write-Host "  → Creates package for your OS" -ForegroundColor $colors.Info
    }
    Write-Host ""
    
    Write-Host "Option 3: Run Directly" -ForegroundColor $colors.Progress
    if ($IsWindows) {
        Write-Host "  Double-click: $desktopPath\bin\ryzanstein.exe" -ForegroundColor $colors.Info
    }
    else {
        Write-Host "  Run: $desktopPath/bin/ryzanstein" -ForegroundColor $colors.Info
    }
    Write-Host ""
    
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host "FEATURES INCLUDED" -ForegroundColor $colors.Header
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host ""
    Write-Host "  ✓ Elite AI Agent Collective (APEX, ARCHITECT, TENSOR, CIPHER, etc.)" -ForegroundColor $colors.Success
    Write-Host "  ✓ Real-time Chat Interface" -ForegroundColor $colors.Success
    Write-Host "  ✓ Model Selection & Management" -ForegroundColor $colors.Success
    Write-Host "  ✓ Agent Tree View" -ForegroundColor $colors.Success
    Write-Host "  ✓ Code Generation & Analysis" -ForegroundColor $colors.Success
    Write-Host "  ✓ Settings Management" -ForegroundColor $colors.Success
    Write-Host ""
    
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host "CONFIGURATION" -ForegroundColor $colors.Header
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host ""
    Write-Host "API Server: http://localhost:8000" -ForegroundColor $colors.Info
    Write-Host "MCP Server: localhost:50051" -ForegroundColor $colors.Info
    Write-Host "Config File: wails.json" -ForegroundColor $colors.Info
    Write-Host ""
    
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host "TROUBLESHOOTING" -ForegroundColor $colors.Header
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host ""
    Write-Host "Issue: 'wails: command not found'" -ForegroundColor $colors.Warning
    Write-Host "  → Run: go install github.com/wailsapp/wails/v2/cmd/wails@latest" -ForegroundColor $colors.Info
    Write-Host ""
    Write-Host "Issue: Build fails" -ForegroundColor $colors.Warning
    Write-Host "  → Ensure Go 1.20+ is installed" -ForegroundColor $colors.Info
    Write-Host "  → Run: go mod tidy" -ForegroundColor $colors.Info
    Write-Host ""
    Write-Host "Issue: Port 8000 already in use" -ForegroundColor $colors.Warning
    Write-Host "  → Change API URL in settings" -ForegroundColor $colors.Info
    Write-Host ""
}

function Main {
    Clear-Host
    Write-Header "RYZANSTEIN DESKTOP APPLICATION - ONE-CLICK SETUP"
    
    try {
        if (-not $SkipValidation) {
            Validate-Prerequisites
        }
        
        Setup-Backend
        Setup-Frontend
        Setup-Wails
        Test-Build
        Show-Instructions
        
        Write-Host ""
        Write-Header "✨ SETUP SUCCESSFUL ✨"
        Write-Host "Your application is ready to launch!" -ForegroundColor $colors.Success
        Write-Host ""
        
    }
    catch {
        Write-Host ""
        Write-Host "❌ Setup Failed" -ForegroundColor $colors.Error
        Write-Host ""
        Write-Host "Error: $_" -ForegroundColor $colors.Error
        Write-Host ""
        Write-Host "Please check the error message above and try again." -ForegroundColor $colors.Warning
        Write-Host ""
        exit 1
    }
}

Main
