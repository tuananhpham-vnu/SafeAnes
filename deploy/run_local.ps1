# SafeAnes Monitor, local preview: one process serves the UI and the API at http://127.0.0.1:8000
#   powershell -ExecutionPolicy Bypass -File deploy\run_local.ps1            # build the UI if missing, then serve
#   powershell -ExecutionPolicy Bypass -File deploy\run_local.ps1 -Rebuild   # rebuild the UI first
# Engine "replay": the probabilities computed by the training run (bundle reference/) are replayed; no model runs on
# this machine (CLAUDE.md). The live engine runs in the container (deploy/Dockerfile) or on Kaggle (check_engine.py).
param([switch]$Rebuild, [int]$Port = 8000)
$ErrorActionPreference = "Continue"
$Deploy = $PSScriptRoot
$Repo = Split-Path $Deploy -Parent
$Backend = Join-Path $Deploy "backend"
$Frontend = Join-Path $Deploy "frontend"
$LocalDeps = Join-Path $Repo ".local_deps"

if (-not (Test-Path (Join-Path $Backend "data_ward\ward.json"))) {
    throw "No bundle in deploy\backend\data_ward: run deploy\backend\export_ward.py first (see deploy\README.md)"
}
$env:PYTHONIOENCODING = "utf-8"
python -c "import fastapi, uvicorn, pandas, pyarrow" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    if (Test-Path $LocalDeps) { $env:PYTHONPATH = $LocalDeps }
    python -c "import fastapi, uvicorn, pandas, pyarrow" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { python -m pip install -q fastapi "uvicorn[standard]" pandas pyarrow }
}

if ($Rebuild -or -not (Test-Path (Join-Path $Frontend "dist\index.html"))) {
    Write-Host "Building the UI..."
    Push-Location $Frontend
    if (-not (Test-Path "node_modules")) { npm install --no-audit --no-fund }
    npm run build
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { throw "UI build failed" }
}

$env:SAFEANES_ENGINE = "replay"
Write-Host "SafeAnes Monitor (replay) on http://127.0.0.1:$Port"
Push-Location $Backend
python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
Pop-Location
