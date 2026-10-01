# Setup script for Story Continuation Generator (FastAPI + React + LSTM)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Setting up Story Continuation Generator Environment..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "Found Python virtual environment in .venv" -ForegroundColor Green
    & .venv\Scripts\python.exe -m pip install -r backend/requirements.txt
} else {
    Write-Host "Creating virtual environment .venv..." -ForegroundColor Yellow
    python -m venv .venv
    & .venv\Scripts\python.exe -m pip install --upgrade pip
    & .venv\Scripts\python.exe -m pip install -r backend/requirements.txt
}

# 2. Check Node & Frontend
if (Test-Path "frontend/package.json") {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    Push-Location frontend
    npm install
    npm run build
    Pop-Location
    Write-Host "Frontend compiled to frontend/dist" -ForegroundColor Green
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "Run the server: .venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Green
