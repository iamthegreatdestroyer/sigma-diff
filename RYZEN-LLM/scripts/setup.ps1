# Ryzanstein LLM Environment Setup Script (Windows/PowerShell)
# [REF:AP-009] - Appendix: Technical Stack

$ErrorActionPreference = "Stop"

Write-Host "=== Ryzanstein LLM Environment Setup (Windows) ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $version = [version]$matches[1]
        if ($version -ge [version]"3.11") {
            Write-Host "✓ $pythonVersion" -ForegroundColor Green
        }
        else {
            Write-Host "✗ Python $version found, but 3.11+ required" -ForegroundColor Red
            exit 1
        }
    }
}
catch {
    Write-Host "✗ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check CMake
Write-Host "Checking CMake..." -ForegroundColor Yellow
try {
    $cmakeVersion = cmake --version 2>&1 | Select-String "cmake version" | ForEach-Object { $_.Line }
    Write-Host "✓ $cmakeVersion" -ForegroundColor Green
}
catch {
    Write-Host "! CMake not found - C++ components will not build" -ForegroundColor Yellow
    Write-Host "  Install from: https://cmake.org/download/" -ForegroundColor Yellow
}

# Check Ninja
Write-Host "Checking Ninja build system..." -ForegroundColor Yellow
try {
    $ninjaVersion = ninja --version 2>&1
    Write-Host "✓ Ninja $ninjaVersion" -ForegroundColor Green
}
catch {
    Write-Host "! Ninja not found - install for faster builds" -ForegroundColor Yellow
    Write-Host "  Install via: choco install ninja" -ForegroundColor Yellow
}

# Check C++ compiler (MSVC or Clang)
Write-Host "Checking C++ compiler..." -ForegroundColor Yellow
$compilerFound = $false

# Check for Visual Studio
try {
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $vsPath = & $vsWhere -latest -property installationPath
        if ($vsPath) {
            Write-Host "✓ Visual Studio found at: $vsPath" -ForegroundColor Green
            $compilerFound = $true
        }
    }
}
catch {
    # VS not found via vswhere
}

# Check for Clang
try {
    $clangVersion = clang --version 2>&1 | Select-Object -First 1
    Write-Host "✓ $clangVersion" -ForegroundColor Green
    $compilerFound = $true
}
catch {
    # Clang not found
}

if (-not $compilerFound) {
    Write-Host "! No C++ compiler found" -ForegroundColor Yellow
    Write-Host "  Install Visual Studio 2022 or Clang" -ForegroundColor Yellow
}

# Check CPU features
Write-Host ""
Write-Host "=== Checking CPU Features ===" -ForegroundColor Cyan
Write-Host ""

$cpuInfo = Get-WmiObject -Class Win32_Processor | Select-Object -First 1
Write-Host "CPU: $($cpuInfo.Name)" -ForegroundColor White

# Check for AVX-512 (requires external tool or manual verification)
Write-Host "! AVX-512 detection requires CPU-Z or manual verification" -ForegroundColor Yellow
Write-Host "  For AMD Ryzanstein 7000+ (Zen 4), AVX-512 should be supported" -ForegroundColor Yellow

# Install Python dependencies
Write-Host ""
Write-Host "=== Installing Python Dependencies ===" -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "Installing core dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing Ryzanstein LLM package..." -ForegroundColor Yellow
python -m pip install -e .

Write-Host "Installing development dependencies..." -ForegroundColor Yellow
python -m pip install -e ".[dev]"

Write-Host "Installing benchmark dependencies..." -ForegroundColor Yellow
python -m pip install -e ".[benchmark]"

Write-Host ""
Write-Host "✓ Python dependencies installed" -ForegroundColor Green

# Create build directory
Write-Host ""
Write-Host "=== Setting Up Build Directory ===" -ForegroundColor Cyan
Write-Host ""

$buildDir = Join-Path $projectRoot "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
    Write-Host "✓ Created build directory" -ForegroundColor Green
}
else {
    Write-Host "✓ Build directory exists" -ForegroundColor Green
}

# Configure CMake (if available)
if (Get-Command cmake -ErrorAction SilentlyContinue) {
    Write-Host ""
    Write-Host "=== Configuring CMake ===" -ForegroundColor Cyan
    Write-Host ""
    
    Set-Location $buildDir
    
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        Write-Host "Using Ninja generator..." -ForegroundColor Yellow
        cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
    }
    else {
        Write-Host "Using default generator..." -ForegroundColor Yellow
        cmake -DCMAKE_BUILD_TYPE=Release ..
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ CMake configuration successful" -ForegroundColor Green
    }
    else {
        Write-Host "✗ CMake configuration failed" -ForegroundColor Red
    }
    
    Set-Location $projectRoot
}

# Create necessary directories
Write-Host ""
Write-Host "=== Creating Storage Directories ===" -ForegroundColor Cyan
Write-Host ""

$directories = @(
    "models",
    "storage/embeddings",
    "storage/kv_cache",
    "storage/rsu_bank"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $projectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "✓ Created $dir" -ForegroundColor Green
    }
    else {
        Write-Host "✓ $dir exists" -ForegroundColor Green
    }
}

# Summary
Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Verify AVX-512 support: python -c `"import cpuinfo; print('avx512f' in cpuinfo.get_cpu_info()['flags'])`"" -ForegroundColor White
Write-Host "  2. Download models: python scripts/download_models.py --model bitnet-1b" -ForegroundColor White
Write-Host "  3. Build C++ components: cd build && ninja (or cmake --build .)" -ForegroundColor White
Write-Host "  4. Run tests: pytest tests/" -ForegroundColor White
Write-Host "  5. Start server: python -m uvicorn src.api.server:app --reload" -ForegroundColor White
Write-Host ""
