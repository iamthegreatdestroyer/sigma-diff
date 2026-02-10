# RYZEN-LLM Integration Test Runner (PowerShell)
# [REF:PHASE1-TASK5] - Automated Testing

param(
    [switch]$Build,
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Release",
    [switch]$Unit,
    [switch]$Integration,
    [switch]$Benchmark,
    [switch]$All,
    [string]$Markers = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "RYZEN-LLM Integration Test Suite" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if it exists
$VenvPath = Join-Path $ProjectRoot ".venv"
if (Test-Path $VenvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    $ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    if (Test-Path $ActivateScript) {
        & $ActivateScript
    }
}

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
$PythonVersion = python --version 2>&1
Write-Host "  $PythonVersion" -ForegroundColor Green

# Check pytest
Write-Host "Checking pytest..." -ForegroundColor Yellow
try {
    python -m pytest --version | Out-Null
    Write-Host "  pytest installed" -ForegroundColor Green
}
catch {
    Write-Host "  Installing pytest..." -ForegroundColor Yellow
    python -m pip install pytest pytest-cov pytest-benchmark numpy
}

# Build if requested
if ($Build) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "Building C++ Project ($Config)" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host ""
    
    $BuildDir = Join-Path $ProjectRoot "build"
    
    # Configure
    Write-Host "Configuring with CMake..." -ForegroundColor Yellow
    cmake -S $ProjectRoot -B $BuildDir -G Ninja "-DCMAKE_BUILD_TYPE=$Config"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed" -ForegroundColor Red
        exit 1
    }
    
    # Build
    Write-Host "Building..." -ForegroundColor Yellow
    cmake --build $BuildDir --config $Config --parallel
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Build successful" -ForegroundColor Green
}

# Default to all tests if none specified
if (-not ($Unit -or $Integration -or $Benchmark)) {
    $All = $true
}

$TestsFailed = $false

# Run Integration Tests
if ($All -or $Integration) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "Running Integration Tests" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host ""
    
    $TestPath = Join-Path $ProjectRoot "tests\integration"
    $PytestArgs = @(
        "-m", "pytest",
        $TestPath,
        "-v",
        "--tb=short",
        "-s",
        "--color=yes",
        "--cov=src",
        "--cov-report=term-missing"
    )
    
    if ($Markers) {
        $PytestArgs += "-m"
        $PytestArgs += $Markers
    }
    
    & python @PytestArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Integration tests failed" -ForegroundColor Red
        $TestsFailed = $true
    }
    else {
        Write-Host ""
        Write-Host "Integration tests passed" -ForegroundColor Green
    }
}

# Run Benchmarks
if ($All -or $Benchmark) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "Running Performance Benchmarks" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host ""
    
    $TestPath = Join-Path $ProjectRoot "tests\integration"
    & python -m pytest $TestPath -v -m benchmark --tb=short
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Benchmarks failed" -ForegroundColor Red
        $TestsFailed = $true
    }
    else {
        Write-Host ""
        Write-Host "Benchmarks completed" -ForegroundColor Green
    }
}

# Summary
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
if ($TestsFailed) {
    Write-Host "FAILED - Some tests did not pass" -ForegroundColor Red
    Write-Host "=" * 80 -ForegroundColor Cyan
    exit 1
}
else {
    Write-Host "SUCCESS - All tests passed" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Cyan
    exit 0
}
