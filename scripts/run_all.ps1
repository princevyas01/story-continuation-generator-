# Run all services: builds frontend and runs single-server FastAPI on port 8000
param (
    [switch]$RebuildFrontend,
    [int]$Port = 8000
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting Story Continuation Generator (Single Server)..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Build frontend if needed
if ($RebuildFrontend -or (-not (Test-Path "frontend/dist/index.html"))) {
    Write-Host "Building React frontend to frontend/dist..." -ForegroundColor Yellow
    Push-Location frontend
    npm run build
    Pop-Location
}

# 2. Run FastAPI Server
$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Application is live at: http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "Swagger API Docs at:     http://127.0.0.1:$Port/docs" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""

& $pythonExe -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port
