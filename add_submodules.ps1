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
    Write-Host ">>> Adding submodule: $dep" -NoNewline
    
    $url = "https://github.com/$org/$dep.git"
    $path = "dependencies/$dep"
    
    $result = git submodule add $url $path 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        $success++
        Write-Host " OK" -ForegroundColor Green
    } else {
        $failed += $dep
        Write-Host " FAIL - $result" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Result: $success/18 succeeded, $($failed.Count) failed"
if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed -join ', ')"
}
