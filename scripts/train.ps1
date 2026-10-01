# Training script for Story Continuation LSTM model
param (
    [switch]$Smoke,
    [int]$Epochs,
    [int]$BatchSize
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting Story Continuation LSTM Model Training..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$cmdArgs = @("ml/src/train.py")
if ($Smoke) {
    $cmdArgs += "--smoke"
}
if ($Epochs -gt 0) {
    $cmdArgs += "--epochs"
    $cmdArgs += $Epochs
}
if ($BatchSize -gt 0) {
    $cmdArgs += "--batch-size"
    $cmdArgs += $BatchSize
}

& $pythonExe $cmdArgs
