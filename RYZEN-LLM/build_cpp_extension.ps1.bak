#!/usr/bin/env pwsh
# RYZEN-LLM C++ Extension Build Script
# This script reproduces the complete build environment setup and compilation

param(
    [string]$BuildType = "Release",
    [int]$Parallel = 4,
    [switch]$Clean = $false,
    [switch]$Verbose = $false
)

# Color output
$Green = @{ ForegroundColor = 'Green' }
$Red = @{ ForegroundColor = 'Red' }
$Yellow = @{ ForegroundColor = 'Yellow' }
$Cyan = @{ ForegroundColor = 'Cyan' }

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    $Color = $Green
    if ($Status -eq "ERROR") { $Color = $Red }
    elseif ($Status -eq "WARN") { $Color = $Yellow }
    elseif ($Status -eq "INFO") { $Color = $Cyan }
    
    Write-Host "[$Status] $Message" @Color
}

function Check-Tool {
    param([string]$Name, [string]$Command)
    
    try {
        $result = & $Command 2>&1
        Write-Status "✅ $Name is available" "INFO"
        return $true
    }
    catch {
        Write-Status "❌ $Name not found: $_" "ERROR"
        return $false
    }
}

# ============================================================================
# MAIN BUILD SCRIPT
# ============================================================================

Write-Status "╔════════════════════════════════════════════════════════════════╗" "INFO"
Write-Status "║  RYZEN-LLM C++ Extension Build Script                         ║" "INFO"
Write-Status "║  Build Type: $BuildType                                    ║" "INFO"
Write-Status "║  Parallel Jobs: $Parallel                                        ║" "INFO"
Write-Status "╚════════════════════════════════════════════════════════════════╝" "INFO"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Status "Working directory: $(Get-Location)" "INFO"

# ============================================================================
# Step 1: Verify Build Tools
# ============================================================================

Write-Status "╔═ STEP 1: Verifying Build Tools ════════════════════════════════╗" "INFO"

# Check Visual Studio
$VSPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
if (Test-Path "$VSPath\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe") {
    Write-Status "✅ Visual Studio 2019 BuildTools found" "INFO"
    $ClPath = "$VSPath\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"
}
else {
    Write-Status "❌ Visual Studio 2019 BuildTools not found at $VSPath" "ERROR"
    exit 1
}

# Check CMake
try {
    $CMakeVersion = cmake --version 2>&1 | Select-Object -First 1
    Write-Status "✅ CMake found: $CMakeVersion" "INFO"
}
catch {
    Write-Status "❌ CMake not found. Install with: pip install cmake" "ERROR"
    exit 1
}

# Check Python
try {
    $PythonVersion = python --version 2>&1
    Write-Status "✅ Python found: $PythonVersion" "INFO"
    $PythonExe = (python -c "import sys; print(sys.executable)") 2>&1
    Write-Status "   Path: $PythonExe" "INFO"
}
catch {
    Write-Status "❌ Python not found" "ERROR"
    exit 1
}

# ============================================================================
# Step 2: Configure Build Environment
# ============================================================================

Write-Status "╔═ STEP 2: Configuring Build Environment ═══════════════════════╗" "INFO"

# Set up Visual Studio environment
$env:VSINSTALLDIR = $VSPath
$env:VCToolsInstallDir = "$VSPath\VC\Tools\MSVC\14.29.30133\"
$env:PATH = "$VSPath\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64;$env:PATH"

# Windows SDK
$SDKPath = "C:\Program Files (x86)\Windows Kits\10"
if (Test-Path $SDKPath) {
    $env:INCLUDE = "$SDKPath\Include\10.0.19041.0\um;$SDKPath\Include\10.0.19041.0\shared;$env:INCLUDE"
    $env:LIB = "$SDKPath\Lib\10.0.19041.0\um\x64;$SDKPath\Lib\10.0.19041.0\ucrt\x64;$env:LIB"
    Write-Status "✅ Windows SDK environment configured" "INFO"
}
else {
    Write-Status "⚠️  Windows SDK not found at $SDKPath" "WARN"
}

Write-Status "✅ Build environment configured" "INFO"

# ============================================================================
# Step 3: Clean Build (Optional)
# ============================================================================

if ($Clean) {
    Write-Status "╔═ STEP 3: Cleaning Previous Build ═════════════════════════════╗" "INFO"
    
    if (Test-Path "build\cpp") {
        Write-Status "Removing build\cpp directory..." "INFO"
        Remove-Item -Recurse -Force "build\cpp" 2>&1 | Out-Null
        Write-Status "✅ Build directory cleaned" "INFO"
    }
    
    if (Test-Path "build\python") {
        Write-Status "Removing build\python directory..." "INFO"
        Remove-Item -Recurse -Force "build\python" 2>&1 | Out-Null
        Write-Status "✅ Python build directory cleaned" "INFO"
    }
}

# ============================================================================
# Step 4: CMake Configuration
# ============================================================================

Write-Status "╔═ STEP 4: CMake Configuration ══════════════════════════════════╗" "INFO"

# Create build directory
if (!(Test-Path "build\cpp")) {
    mkdir "build\cpp" 2>&1 | Out-Null
    Write-Status "Created build/cpp directory" "INFO"
}

# Configure CMake
Set-Location "build\cpp"

$CMakeCmd = @(
    "cmake",
    "..\.",
    "-G", "Visual Studio 16 2019",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DPYTHON_EXECUTABLE=$PythonExe",
    "-DPYTHON_INCLUDE_DIR=$(python -c 'import sysconfig; print(sysconfig.get_path(\"include\"))' 2>&1)",
    "-DPYTHON_LIBRARY=$(python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDEST\"))' 2>&1)",
    "-DBUILD_SHARED_LIBS=ON"
)

Write-Status "Running CMake configure..." "INFO"
if ($Verbose) {
    & cmake @CMakeCmd
}
else {
    & cmake @CMakeCmd 2>&1 | Select-Object -Last 20
}

if ($LASTEXITCODE -eq 0) {
    Write-Status "✅ CMake configuration successful" "INFO"
}
else {
    Write-Status "❌ CMake configuration failed" "ERROR"
    exit 1
}

# ============================================================================
# Step 5: Build
# ============================================================================

Write-Status "╔═ STEP 5: Building C++ Extension ═══════════════════════════════╗" "INFO"

Write-Status "Running CMake build (parallel: $Parallel)..." "INFO"

$BuildCmd = @(
    "cmake",
    "--build", ".",
    "--config", $BuildType,
    "--parallel", $Parallel.ToString()
)

if ($Verbose) {
    & cmake @BuildCmd
}
else {
    & cmake @BuildCmd 2>&1
}

if ($LASTEXITCODE -eq 0) {
    Write-Status "✅ Build successful!" "INFO"
}
else {
    Write-Status "❌ Build failed" "ERROR"
    exit 1
}

Set-Location $ScriptDir

# ============================================================================
# Step 6: Verify Output
# ============================================================================

Write-Status "╔═ STEP 6: Verifying Build Output ═══════════════════════════════╗" "INFO"

$PydFile = "build\python\ryzen_llm\ryzen_llm_bindings.pyd"
if (Test-Path $PydFile) {
    $FileSize = (Get-Item $PydFile).Length
    Write-Status "✅ Extension module found: $PydFile ($FileSize bytes)" "INFO"
}
else {
    Write-Status "❌ Extension module not found: $PydFile" "ERROR"
    exit 1
}

# ============================================================================
# Step 7: Run Validation Test
# ============================================================================

Write-Status "╔═ STEP 7: Running Extension Validation ══════════════════════════╗" "INFO"

if (Test-Path "test_extension_load.py") {
    Write-Status "Running extension validation test..." "INFO"
    python test_extension_load.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Status "✅ Extension validation passed!" "INFO"
    }
    else {
        Write-Status "⚠️  Extension validation returned non-zero exit code" "WARN"
    }
}
else {
    Write-Status "⚠️  Validation test not found: test_extension_load.py" "WARN"
}

# ============================================================================
# Build Complete
# ============================================================================

Write-Status "╔════════════════════════════════════════════════════════════════╗" "INFO"
Write-Status "║  BUILD COMPLETE ✅                                             ║" "INFO"
Write-Status "║                                                                ║" "INFO"
Write-Status "║  Output Location:                                              ║" "INFO"
Write-Status "║  $PydFile" "INFO"
Write-Status "║                                                                ║" "INFO"
Write-Status "║  To use the extension in Python:                               ║" "INFO"
Write-Status "║  import sys                                                    ║" "INFO"
Write-Status "║  sys.path.insert(0, 'build/python')                            ║" "INFO"
Write-Status "║  import ryzen_llm.ryzen_llm_bindings                           ║" "INFO"
Write-Status "║                                                                ║" "INFO"
Write-Status "╚════════════════════════════════════════════════════════════════╝" "INFO"

exit 0
