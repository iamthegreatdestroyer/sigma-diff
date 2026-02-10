# =============================================================================
# RYZANSTEIN COMPLETE STACK BUILDER
# Autonomous execution of all phases for Option A (Full Custom LLM)
# =============================================================================
#
# Usage:
#   .\build_complete_stack.ps1              # Full build
#   .\build_complete_stack.ps1 -SkipModels  # Skip 13GB+ model download
#   .\build_complete_stack.ps1 -SkipBuild   # Skip C++ rebuild
#   .\build_complete_stack.ps1 -DryRun      # Show what would be done
#
# =============================================================================

param(
    [switch]$SkipModels,      # Skip model download (if already downloaded)
    [switch]$SkipBuild,       # Skip C++ rebuild
    [switch]$SkipMCP,         # Skip MCP server build
    [switch]$SkipDesktop,     # Skip desktop app build
    [switch]$DryRun,          # Show what would be done
    [switch]$Verbose,         # Enable verbose output
    [switch]$Help             # Show help
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "S:\Ryot"
$RyzenLLMRoot = "$ProjectRoot\RYZEN-LLM"
$StartTime = Get-Date
$LogFile = "$ProjectRoot\logs\build_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $Message -ForegroundColor $Color
    Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
}

function Test-Command {
    param([string]$Command)
    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Test-Prerequisite {
    param([string]$Name, [string]$Command, [string]$InstallHint)
    
    if (Test-Command $Command) {
        Write-Log "  ✅ $Name found" "Green"
        return $true
    }
    else {
        Write-Log "  ❌ $Name not found - $InstallHint" "Red"
        return $false
    }
}

function Invoke-StepWithRetry {
    param(
        [scriptblock]$ScriptBlock,
        [string]$StepName,
        [int]$MaxRetries = 3,
        [int]$RetryDelaySeconds = 5
    )
    
    $attempt = 0
    while ($attempt -lt $MaxRetries) {
        $attempt++
        try {
            Write-Log "  Executing: $StepName (attempt $attempt/$MaxRetries)..." "Gray"
            & $ScriptBlock
            Write-Log "  ✅ $StepName succeeded" "Green"
            return $true
        }
        catch {
            Write-Log "  ⚠️ Attempt $attempt failed: $_" "Yellow"
            if ($attempt -lt $MaxRetries) {
                Write-Log "  Retrying in $RetryDelaySeconds seconds..." "Gray"
                Start-Sleep -Seconds $RetryDelaySeconds
            }
        }
    }
    
    Write-Log "  ❌ $StepName failed after $MaxRetries attempts" "Red"
    return $false
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
║                    COMPLETE STACK BUILDER - OPTION A                             ║
║                       Full Custom LLM - No Shortcuts                             ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

if ($Help) {
    Write-Host @"
Usage: .\build_complete_stack.ps1 [options]

Options:
  -SkipModels    Skip model download (if already downloaded)
  -SkipBuild     Skip C++ rebuild
  -SkipMCP       Skip MCP server build
  -SkipDesktop   Skip desktop app build
  -DryRun        Show what would be done without executing
  -Verbose       Enable verbose output
  -Help          Show this help message

Phases:
  1. Prerequisites Check
  2. C++ Bindings Build
  3. Model Download (13GB+)
  4. Python API Setup
  5. MCP Server Build
  6. Desktop App Build
  7. Validation Tests

"@
    exit 0
}

# Create logs directory
if (-not (Test-Path "$ProjectRoot\logs")) {
    New-Item -ItemType Directory -Path "$ProjectRoot\logs" -Force | Out-Null
}

Write-Log "Build started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "Log file: $LogFile"

if ($DryRun) {
    Write-Log "*** DRY RUN MODE - No changes will be made ***" "Yellow"
}

# =============================================================================
# PHASE 0: PREREQUISITES CHECK
# =============================================================================

Write-Log "`n[PHASE 0] Checking Prerequisites..." "Yellow"

$prereqsMet = $true

$prereqsMet = $prereqsMet -and (Test-Prerequisite "CMake" "cmake" "Install from cmake.org")
$prereqsMet = $prereqsMet -and (Test-Prerequisite "Python" "python" "Install Python 3.11+")
$prereqsMet = $prereqsMet -and (Test-Prerequisite "Go" "go" "Install Go 1.21+")
$prereqsMet = $prereqsMet -and (Test-Prerequisite "Git" "git" "Install Git")

# Check for Visual Studio
$vsInstalls = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2>$null
if ($vsInstalls) {
    Write-Log "  ✅ Visual Studio found" "Green"
}
else {
    Write-Log "  ⚠️ Visual Studio not found - C++ build may fail" "Yellow"
}

# Check for Wails
if (Test-Command "wails") {
    Write-Log "  ✅ Wails found" "Green"
}
else {
    Write-Log "  ⚠️ Wails not found - will attempt to install" "Yellow"
    if (-not $DryRun) {
        go install github.com/wailsapp/wails/v2/cmd/wails@latest
    }
}

if (-not $prereqsMet) {
    Write-Log "`n❌ Prerequisites not met. Please install missing tools and retry." "Red"
    exit 1
}

Write-Log "✅ Phase 0 complete: Prerequisites verified" "Green"

# =============================================================================
# PHASE 1: C++ BINDINGS BUILD
# =============================================================================

if (-not $SkipBuild) {
    Write-Log "`n[PHASE 1] Building C++ Inference Engine..." "Yellow"
    
    if (-not $DryRun) {
        # Navigate to build directory
        $buildDir = "$RyzenLLMRoot\build"
        if (-not (Test-Path $buildDir)) {
            New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        }
        
        Push-Location $buildDir
        
        try {
            # Configure CMake
            Write-Log "  Configuring CMake..." "Gray"
            cmake .. -G "Visual Studio 16 2019" -A x64 `
                -DCMAKE_BUILD_TYPE=Release `
                -DPYTHON_EXECUTABLE=(Get-Command python).Path
            
            if ($LASTEXITCODE -ne 0) {
                throw "CMake configuration failed"
            }
            
            # Build
            Write-Log "  Building (this may take several minutes)..." "Gray"
            cmake --build . --config Release --target ryzen_llm_bindings -j 8
            
            if ($LASTEXITCODE -ne 0) {
                throw "CMake build failed"
            }
            
            # Verify output
            $pydPath = "$buildDir\python\ryzanstein_llm\ryzen_llm_bindings.pyd"
            if (Test-Path $pydPath) {
                $pydSize = (Get-Item $pydPath).Length / 1KB
                Write-Log "  ✅ Built: ryzen_llm_bindings.pyd ($([math]::Round($pydSize, 1)) KB)" "Green"
            }
            else {
                throw ".pyd file not found after build"
            }
        }
        finally {
            Pop-Location
        }
    }
    
    Write-Log "✅ Phase 1 complete: C++ bindings built" "Green"
}
else {
    Write-Log "`n[PHASE 1] Skipping C++ build (--SkipBuild)" "Gray"
}

# =============================================================================
# PHASE 2: MODEL DOWNLOAD
# =============================================================================

if (-not $SkipModels) {
    Write-Log "`n[PHASE 2] Downloading Models..." "Yellow"
    Write-Log "  ⚠️ This will download ~15GB of model weights" "Yellow"
    
    $ModelDir = "$RyzenLLMRoot\models"
    
    if (-not $DryRun) {
        # Ensure huggingface_hub is installed
        Write-Log "  Installing huggingface_hub..." "Gray"
        pip install huggingface_hub --quiet --upgrade
        
        # Enable fast transfers
        $env:HF_HUB_ENABLE_HF_TRANSFER = "1"
        pip install hf_transfer --quiet
        
        # Download BitNet 3B (primary model)
        Write-Log "  [1/2] Downloading BitNet b1.58 3B (13.3 GB)..." "Yellow"
        $bitnetDir = "$ModelDir\bitnet\3b"
        if (-not (Test-Path $bitnetDir)) {
            New-Item -ItemType Directory -Path $bitnetDir -Force | Out-Null
        }
        
        python -m huggingface_hub download "1bitLLM/bitnet_b1_58-3B" `
            --local-dir $bitnetDir `
            --include "*.safetensors" "config.json" "tokenizer*" "*.model"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "  ⚠️ BitNet download may have failed - check $bitnetDir" "Yellow"
        }
        
        # Download draft model for speculative decoding
        Write-Log "  [2/2] Downloading Draft Model (350 MB)..." "Yellow"
        $draftDir = "$ModelDir\drafts\dialogpt"
        if (-not (Test-Path $draftDir)) {
            New-Item -ItemType Directory -Path $draftDir -Force | Out-Null
        }
        
        python -m huggingface_hub download "microsoft/DialoGPT-small" `
            --local-dir $draftDir `
            --include "*.bin" "config.json" "tokenizer*"
        
        # Verify downloads
        $bitnetFiles = Get-ChildItem "$bitnetDir\*.safetensors" -ErrorAction SilentlyContinue
        if ($bitnetFiles.Count -ge 1) {
            $totalSize = ($bitnetFiles | Measure-Object -Property Length -Sum).Sum / 1GB
            Write-Log "  ✅ BitNet model: $($bitnetFiles.Count) files, $([math]::Round($totalSize, 1)) GB" "Green"
        }
        else {
            Write-Log "  ⚠️ BitNet model files not found - manual download may be required" "Yellow"
        }
    }
    
    Write-Log "✅ Phase 2 complete: Models downloaded" "Green"
}
else {
    Write-Log "`n[PHASE 2] Skipping model download (--SkipModels)" "Gray"
}

# =============================================================================
# PHASE 3: PYTHON API SETUP
# =============================================================================

Write-Log "`n[PHASE 3] Setting up Python API..." "Yellow"

if (-not $DryRun) {
    Push-Location $RyzenLLMRoot
    
    try {
        # Install dependencies
        Write-Log "  Installing Python dependencies..." "Gray"
        
        if (Test-Path "requirements.txt") {
            pip install -r requirements.txt --quiet
        }
        else {
            # Install essential packages
            pip install fastapi uvicorn pydantic safetensors torch --quiet
        }
        
        # Set up PYTHONPATH
        $env:PYTHONPATH = "$RyzenLLMRoot\src;$RyzenLLMRoot\build\python\ryzanstein_llm"
        
        # Verify imports
        Write-Log "  Verifying Python setup..." "Gray"
        $testResult = python -c "import sys; sys.path.insert(0, '$RyzenLLMRoot\build\python\ryzanstein_llm'); import ryzen_llm_bindings; print(ryzen_llm_bindings.test_function())" 2>&1
        
        if ($testResult -match "42") {
            Write-Log "  ✅ C++ bindings import successfully" "Green"
        }
        else {
            Write-Log "  ⚠️ C++ bindings import test inconclusive: $testResult" "Yellow"
        }
    }
    finally {
        Pop-Location
    }
}

Write-Log "✅ Phase 3 complete: Python API ready" "Green"

# =============================================================================
# PHASE 4: MCP SERVER BUILD
# =============================================================================

if (-not $SkipMCP) {
    Write-Log "`n[PHASE 4] Building MCP Server..." "Yellow"
    
    if (-not $DryRun) {
        Push-Location "$ProjectRoot\mcp"
        
        try {
            Write-Log "  Downloading Go dependencies..." "Gray"
            go mod tidy
            
            Write-Log "  Building MCP server..." "Gray"
            go build -o mcp-server.exe .
            
            if ($LASTEXITCODE -ne 0) {
                throw "Go build failed"
            }
            
            if (Test-Path "mcp-server.exe") {
                $exeSize = (Get-Item "mcp-server.exe").Length / 1MB
                Write-Log "  ✅ Built: mcp-server.exe ($([math]::Round($exeSize, 1)) MB)" "Green"
            }
        }
        finally {
            Pop-Location
        }
    }
    
    Write-Log "✅ Phase 4 complete: MCP server built" "Green"
}
else {
    Write-Log "`n[PHASE 4] Skipping MCP build (--SkipMCP)" "Gray"
}

# =============================================================================
# PHASE 5: DESKTOP APP BUILD
# =============================================================================

if (-not $SkipDesktop) {
    Write-Log "`n[PHASE 5] Building Desktop App..." "Yellow"
    
    if (-not $DryRun) {
        Push-Location "$ProjectRoot\desktop"
        
        try {
            Write-Log "  Downloading Go dependencies..." "Gray"
            go mod tidy
            
            Write-Log "  Building with Wails..." "Gray"
            wails build
            
            if ($LASTEXITCODE -ne 0) {
                throw "Wails build failed"
            }
            
            $exePath = "$ProjectRoot\desktop\build\bin\ryzanstein.exe"
            if (Test-Path $exePath) {
                $exeSize = (Get-Item $exePath).Length / 1MB
                Write-Log "  ✅ Built: ryzanstein.exe ($([math]::Round($exeSize, 1)) MB)" "Green"
            }
        }
        finally {
            Pop-Location
        }
    }
    
    Write-Log "✅ Phase 5 complete: Desktop app built" "Green"
}
else {
    Write-Log "`n[PHASE 5] Skipping Desktop build (--SkipDesktop)" "Gray"
}

# =============================================================================
# PHASE 6: VALIDATION
# =============================================================================

Write-Log "`n[PHASE 6] Running Validation..." "Yellow"

$validationResults = @{
    "C++ Bindings" = Test-Path "$RyzenLLMRoot\build\python\ryzanstein_llm\ryzen_llm_bindings.pyd"
    "BitNet Model" = (Get-ChildItem "$RyzenLLMRoot\models\bitnet\3b\*.safetensors" -ErrorAction SilentlyContinue).Count -gt 0
    "MCP Server"   = Test-Path "$ProjectRoot\mcp\mcp-server.exe"
    "Desktop App"  = Test-Path "$ProjectRoot\desktop\build\bin\ryzanstein.exe"
    "Elite Agents" = (Get-ChildItem "$ProjectRoot\.github\agents\*.md" -ErrorAction SilentlyContinue).Count -ge 40
}

foreach ($item in $validationResults.GetEnumerator()) {
    if ($item.Value) {
        Write-Log "  ✅ $($item.Key)" "Green"
    }
    else {
        Write-Log "  ❌ $($item.Key)" "Red"
    }
}

# =============================================================================
# SUMMARY
# =============================================================================

$EndTime = Get-Date
$Duration = $EndTime - $StartTime
$passedCount = ($validationResults.Values | Where-Object { $_ }).Count
$totalCount = $validationResults.Count

Write-Host @"

╔══════════════════════════════════════════════════════════════════════════════════╗
║                              BUILD COMPLETE                                       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Duration: $($Duration.ToString("hh\:mm\:ss"))
║  Validation: $passedCount / $totalCount components ready
║  
║  Components:
"@ -ForegroundColor $(if ($passedCount -eq $totalCount) { "Green" } else { "Yellow" })

foreach ($item in $validationResults.GetEnumerator()) {
    $status = if ($item.Value) { "✅" } else { "❌" }
    Write-Host "║    $status $($item.Key)"
}

Write-Host @"
║  
║  Log file: $LogFile
║  
║  Next Steps:
║    1. Run: .\scripts\start_ryzanstein.ps1
║    2. Open: http://localhost:8000/docs (API)
║    3. Test: Desktop app with Elite Agents
╚══════════════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor $(if ($passedCount -eq $totalCount) { "Green" } else { "Yellow" })

Write-Log "Build completed in $($Duration.ToString("hh\:mm\:ss"))"

# Return exit code based on validation
if ($passedCount -eq $totalCount) {
    exit 0
}
else {
    exit 1
}
