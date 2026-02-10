# Ryzanstein LLM Test Runner (PowerShell)
# [REF:PHASE1-002] - Unit Test Execution

param(
    [string]$TestPattern = "test_bitnet_*",
    [switch]$Verbose,
    [switch]$Coverage
)

Write-Host "=== Ryzanstein LLM Unit Test Runner ===" -ForegroundColor Cyan

# Navigate to project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# Check pytest is installed
if (-not (Get-Command pytest -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: pytest not found. Installing..." -ForegroundColor Red
    pip install pytest pytest-cov numpy
}

# Build C++ components first
Write-Host "`nBuilding C++ components..." -ForegroundColor Yellow
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Push-Location build

try {
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: CMake configuration failed" -ForegroundColor Red
        exit 1
    }

    ninja
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "Build successful!" -ForegroundColor Green
}
finally {
    Pop-Location
}

# Run Python tests
Write-Host "`nRunning tests matching: $TestPattern" -ForegroundColor Yellow

$PytestArgs = @(
    "tests/unit/$TestPattern.py",
    "-v",
    "--tb=short"
)

if ($Verbose) {
    $PytestArgs += "-s"
}

if ($Coverage) {
    $PytestArgs += @(
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_html"
    )
}

Write-Host "`nCommand: pytest $($PytestArgs -join ' ')" -ForegroundColor Gray

pytest @PytestArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ All tests passed!" -ForegroundColor Green
}
else {
    Write-Host "`n✗ Some tests failed." -ForegroundColor Red
    exit 1
}

if ($Coverage) {
    Write-Host "`nCoverage report generated in: coverage_html/index.html" -ForegroundColor Cyan
}
