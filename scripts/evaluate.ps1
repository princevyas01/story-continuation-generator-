# Evaluation script for Story Continuation LSTM model
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Running Story Continuation LSTM Evaluation Suite..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

& $pythonExe ml/src/evaluate.py
