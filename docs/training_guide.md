# Story Continuation Generator v2 — Training & Reproduction Guide

This guide provides the exact commands and instructions to reproduce data acquisition, subword tokenization, LSTM model training, comparative evaluation, and application deployment from scratch.

---

## 1. Environment Requirements

- **Operating System:** Windows 11 (or Linux/macOS)
- **Processor:** Intel Core Ultra 5 125H (or modern 6+ core x86_64 CPU)
- **Memory:** 16 GB RAM
- **Python Version:** 3.13+ (or 3.10–3.12)
- **Node.js:** v18+ (for building the frontend)

### Virtual Environment Setup

In PowerShell from repository root:

```powershell
# Create virtual environment if not present
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1

# Install core dependencies
pip install tensorflow keras tokenizers datasets sentencepiece pydantic fastapi uvicorn pytest
```

---

## 2. Dataset Preparation (Phase 2)

Acquires a deterministic, buffer-shuffled subset of **5,000 stories** from Hugging Face `roneneldan/TinyStories` combined with **67 curated fairy tales** from Project Gutenberg. Documents are cleaned, stripped of Gutenberg boilerplate, deduplicated by SHA-256 hash, and deterministically split into 90% train, 5% validation, and 5% test sets.

```powershell
$env:PYTHONPATH="."
python scripts/prepare_tinystories.py
```

**Outputs generated:**
- `data/processed/v2/train.jsonl` (4,560 stories, ~824k words, ~1.05M subwords)
- `data/processed/v2/val.jsonl` (253 stories, ~46.6k words, ~58k subwords)
- `data/processed/v2/test.jsonl` (254 stories, ~47.3k words, ~59k subwords)
- `data/manifests/dataset_v2_manifest.json` (Source licenses, token counts, SHA-256 hashes)

---

## 3. Subword Tokenizer Training (Phase 3)

Trains an 8,000-vocabulary Byte-Level BPE tokenizer **strictly on `train.jsonl`** (zero contamination of val or test sets). Assigns explicit special tokens:
- `<PAD>`: ID 0
- `<UNK>`: ID 1
- `<BOS>`: ID 2
- `<EOS>`: ID 3

```powershell
$env:PYTHONPATH="."
python scripts/train_subword_tokenizer.py
```

**Outputs generated:**
- `artifacts/tokenizers/tokenizer_v2.json`
- `artifacts/tokenizers/tokenizer_v2_metadata.json`
- `artifacts/tokenizers/tokenizer_v2_stats.json` (Verifies 0.000% unknown-token rate on val and test)

---

## 4. LSTM Model Training (Phase 4 & 5)

Trains the two-layer stacked LSTM with many-to-many autoregressive supervision across sequences of length 128 tokens, utilizing masked cross-entropy loss that ignores `<PAD>` positions.

### Quick Smoke Test (15 seconds)

```powershell
$env:PYTHONPATH="."
python ml/src/train_v2.py --smoke-test
```

### Full Production Training

```powershell
$env:PYTHONPATH="."
python ml/src/train_v2.py --epochs 8 --batch-size 64 --seq-len 128 --stride 96 --lr 0.0015
```

**Outputs generated:**
- `artifacts/models/story_lstm_v2.keras` (Best validation checkpoint)
- `models/best/story_lstm_v2.keras` (Synced copy for local inference)
- `artifacts/models/model_v2_config.json` (Hyperparameter record)
- `artifacts/models/model_summary_v2.txt` (Architecture details: 4,577,088 parameters)
- `artifacts/reports/training_history_v2.json` (Per-epoch training loss, accuracy, and perplexity)

---

## 5. Comprehensive Comparative Evaluation (Phase 7)

Evaluates the upgraded candidate (v2) head-to-head against the preserved baseline (v1) across:
- Masked test loss, test perplexity, and token accuracy on held-out `test.jsonl`.
- 30 curated prompts spanning 8 diverse narrative categories.
- Distinct-1 and Distinct-2 n-gram vocabulary diversity.
- 4-gram repetition degradation rate.
- Measured CPU inference latency (ms per token).

```powershell
$env:PYTHONPATH="."
python ml/src/evaluate_v2.py
```

**Outputs generated:**
- `artifacts/reports/evaluation_comparison_v2.json`
- `artifacts/reports/human_evaluation_sheet_v2.csv`
- `artifacts/reports/training_report_v2.md`

---

## 6. Interactive CLI Generation

Test interactive generation from the command line:

```powershell
$env:PYTHONPATH="."
python ml/src/generate_cli.py --prompt "Once upon a time, Lily found a golden key in the forest" --tokens 60 --temperature 0.75
```

---

## 7. Running the Fullstack Application

### Step A: Build Frontend

```powershell
cd frontend
npm run build
cd ..
```

### Step B: Launch FastAPI Server

FastAPI serves both the REST API endpoints (`/api/v1/...`) and the compiled React single-page application at root:

```powershell
$env:PYTHONPATH="."
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
`http://127.0.0.1:8000`
