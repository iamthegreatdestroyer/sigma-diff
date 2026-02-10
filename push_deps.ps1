$deps = @(
    "agentmem", "ann-hybrid", "archaeo", "causedb", "cpu-infer", "dep-bloom",
    "flowstate", "intent-spec", "mcp-mesh", "neurectomy-shell", "semlog",
    "sigma-api", "sigma-compress", "sigma-diff", "sigma-index", "sigma-telemetry",
    "vault-git", "zkaudit"
)

$org = "iamthegreatdestroyer"
$success = 0
$failed = @()

foreach ($dep in $deps) {
    $p = "S:\Ryot\dependencies\$dep"
    Write-Host ">>> $dep" -NoNewline
    
    Push-Location $p
    
    if (Test-Path ".git") {
        Remove-Item -Recurse -Force ".git"
    }
    
    git init --initial-branch=main 2>&1 | Out-Null
    git remote add origin "https://github.com/$org/$dep.git" 2>&1 | Out-Null
    git add -A 2>&1 | Out-Null
    git commit -m "feat: initial scaffold for $dep - Ryzanstein ecosystem dependency" 2>&1 | Out-Null
    git push -u origin main --force 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        $success++
        Write-Host " OK" -ForegroundColor Green
    } else {
        $failed += $dep
        Write-Host " FAIL" -ForegroundColor Red
    }
    
    Pop-Location
}

Write-Host ""
Write-Host "Result: $success/18 succeeded, $($failed.Count) failed"
if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed -join ', ')"
}
