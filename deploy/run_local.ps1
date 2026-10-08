# SafeAnes Monitor, local: one process serves the UI and the API at http://127.0.0.1:8000
#   powershell -ExecutionPolicy Bypass -File deploy\run_local.ps1            # export/build only what is missing
#   powershell -ExecutionPolicy Bypass -File deploy\run_local.ps1 -Rebuild   # re-export data and rebuild the UI
param([switch]$Rebuild, [int]$Port = 8000)
# "Continue": in Windows PowerShell 5.1 "Stop" turns any stderr line of a native command into a fatal error;
# native failures are checked with $LASTEXITCODE below instead
$ErrorActionPreference = "Continue"
$Deploy = $PSScriptRoot
$Repo = Split-Path $Deploy -Parent
$Backend = Join-Path $Deploy "backend"
$Frontend = Join-Path $Deploy "frontend"
$LocalDeps = Join-Path $Repo ".local_deps"

# python libraries: lightgbm etc. may live in the repo's .local_deps
$env:PYTHONIOENCODING = "utf-8"
python -c "import fastapi, uvicorn, lightgbm, pandas, pyarrow" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    if (Test-Path $LocalDeps) { $env:PYTHONPATH = $LocalDeps }
    python -c "import fastapi, uvicorn, lightgbm, pandas, pyarrow" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { python -m pip install -q -r (Join-Path $Backend "requirements.txt") }
}

# 1. models + demo cases (needs artifacts/v3 and data/samples_v3)
if ($Rebuild -or -not (Test-Path (Join-Path $Backend "data\cases.json"))) {
    Write-Host "Exporting models and demo cases..."
    $saved = $env:PYTHONPATH
    $env:PYTHONPATH = "$(Join-Path $Repo 'src');$LocalDeps"
    Push-Location $Repo
    python deploy\backend\export_bundle.py --config configs\uc04_v3.json --work artifacts\v3 --cases 14
    $code = $LASTEXITCODE
    Pop-Location
    $env:PYTHONPATH = $saved
    if ($code -ne 0) { throw "export_bundle.py failed" }
}

# 1b. example uploads for the "Dự báo" page + check of the feature extraction (needs data/vitaldb_full raw tracks)
if (($Rebuild -or -not (Test-Path (Join-Path $Backend "data\examples.json"))) -and (Test-Path (Join-Path $Repo "data\vitaldb_full\raw"))) {
    Write-Host "Building example files and checking feature extraction..."
    $saved = $env:PYTHONPATH
    $env:PYTHONPATH = "$(Join-Path $Repo 'src');$LocalDeps;$Backend"
    Push-Location $Repo
    python deploy\backend\check_extract.py
    Pop-Location
    $env:PYTHONPATH = $saved
}

# 2. UI build
if ($Rebuild -or -not (Test-Path (Join-Path $Frontend "dist\index.html"))) {
    Write-Host "Building the UI..."
    Push-Location $Frontend
    if (-not (Test-Path "node_modules")) { npm install --no-audit --no-fund }
    npm run build
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { throw "npm run build failed" }
}

# 3. serve (localhost only)
Write-Host "SafeAnes Monitor: http://127.0.0.1:$Port   (API docs: /docs)   Ctrl+C to stop"
Start-Job -ScriptBlock { param($p) Start-Sleep 3; Start-Process "http://127.0.0.1:$p" } -ArgumentList $Port | Out-Null
Push-Location $Backend
try { python -m uvicorn app.main:app --host 127.0.0.1 --port $Port }
finally { Pop-Location }
