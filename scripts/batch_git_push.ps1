# Batch Git Push Script: Pushes 2 files per commit with a 10-second gap
$ErrorActionPreference = "Stop"

# Get all untracked and modified files
$rawStatus = git status -uall -s
$files = @()
foreach ($line in $rawStatus) {
    if ($line.Length -gt 3) {
        $f = $line.Substring(3).Trim()
        # Clean quotes if present
        $f = $f.Trim('"')
        if (Test-Path $f) {
            $files += $f
        }
    }
}

# Prioritize logical ordering
$priority = @(
    "requirements.txt",
    "configs/config.yaml",
    "src/__init__.py", "src/config.py",
    "src/preprocessing.py", "src/tokenizer.py",
    "src/sequences.py", "src/model.py",
    "src/metrics.py", "src/generate.py",
    "src/train.py", "src/evaluate.py",
    "scripts/check_environment.py", "scripts/prepare_data.py",
    "scripts/smoke_test.py", "app/ui_helpers.py",
    "app/app.py", "tests/test_config.py",
    "tests/test_preprocessing.py", "tests/test_tokenizer.py",
    "tests/test_sequence_creation.py", "tests/test_model_shape.py",
    "tests/test_generation.py", "tests/test_metrics.py",
    "tests/test_artifact_loading.py", "tests/test_smoke.py",
    "docs/ARCHITECTURE.md", "docs/METHODOLOGY.md",
    "docs/TECHNICAL_REFERENCES.md", "docs/VIVA_QA.md",
    "docs/DEMO_SCRIPT.md", "docs/PPT_OUTLINE.md",
    "docs/LIMITATIONS.md", "docs/EXPERIMENT_LOG.md",
    "data/manifests/dataset_manifest.json", "artifacts/model_summary.txt",
    "artifacts/tokenizer_vocab.json", "artifacts/config_used.yaml",
    "artifacts/metrics/metrics.json", "artifacts/metrics/metrics.csv",
    "artifacts/plots/loss_curve.png", "artifacts/reports/training_report.md",
    "artifacts/reports/human_evaluation_sheet.csv", "artifacts/samples/sample_generations.json",
    "artifacts/samples/sample_generations.md", "models/best/story_lstm.keras",
    "models/final/story_lstm_final.keras"
)

$ordered = @()
foreach ($p in $priority) {
    if ($files -contains $p) {
        $ordered += $p
    }
}
foreach ($f in $files) {
    if (-not ($ordered -contains $f)) {
        $ordered += $f
    }
}

Write-Host "Total files to process: $($ordered.Count)"

$commitIndex = 1
for ($i = 0; $i -lt $ordered.Count; $i += 2) {
    $batch = @()
    $batch += $ordered[$i]
    if ($i + 1 -lt $ordered.Count) {
        $batch += $ordered[$i + 1]
    }

    Write-Host "----------------------------------------------------"
    Write-Host "Batch ${commitIndex}: Staging $($batch -join ', ')"

    foreach ($file in $batch) {
        git add -- "$file"
    }

    $commitMsg = "feat: add $($batch -join ' and ')"
    git commit -m "$commitMsg"
    git push origin main

    Write-Host "Batch ${commitIndex} successfully pushed to GitHub."
    $commitIndex++
}

Write-Host "===================================================="
Write-Host "All batches committed and pushed successfully!"
Write-Host "===================================================="
