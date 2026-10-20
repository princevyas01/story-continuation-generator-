# 📖 Story Continuation Generator Using Stacked LSTM

> A production-quality, locally trained **2-Layer LSTM Neural Language Model** paired with a modern **FastAPI** backend and **React + TypeScript + Vite** frontend. Built CPU-first for Windows 11 on the **Intel Core Ultra 5 125H**. Zero external generative APIs.

---

## 🌟 Overview & System Evolution

Language generation often relies on multi-billion parameter cloud LLMs that introduce network latency, vendor lock-in, and unpredictable costs. This project implements a **100% self-contained autoregressive language model** using deep recurrent neural networks:

- **Story LSTM v3 (Active Default):** Trained on 50,000 TinyStories documents with 8,000 Byte-Level BPE subwords, 128 context length, and raw logits outputs. Eliminates repetitive loops and produces coherent, multi-sentence stories.
- **Story LSTM v2 (Preserved Rollback):** Many-to-many autoregressive LSTM with masked cross-entropy. Fully preserved for immediate rollback.
- **Story LSTM v1 (Legacy Baseline):** Word-level many-to-one LSTM with 5,004 vocabulary items.
- **CPU-First Performance:** Optimized for thin-and-light laptop CPUs (Intel Core Ultra 5 125H) without requiring NVIDIA CUDA.
- **Modern Single-Server Stack:** FastAPI backend serving the compiled Vite React frontend on port `8000`.
- **Dynamic Sampling Engine:** Next-token generation featuring Temperature scaling, Top-K truncation, Top-P (Nucleus) sampling, and Repetition Penalties with special-token protection.

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │   React 18 + TypeScript + Vite (Port 8000)   │
                    │   - PromptEditor with Preset Chips           │
                    │   - Dynamic Temperature / Top-K Controls     │
                    │   - Model Version Toggle (v2 vs v3)          │
                    │   - OutputCard with Performance Diagnostics  │
                    └──────────────────────┬───────────────────────┘
                                           │ HTTP POST /api/v1/generate
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │            FastAPI Backend Service           │
                    │   - Pydantic Request/Response Validation     │
                    │   - Thread-safe ModelService Singleton       │
                    │   - Model Registry (v3 Active, v2 Rollback)  │
                    │   - Health & Status Probes                   │
                    │   - Static SPA Catch-all Server              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         LSTM v3 Inference Pipeline                               │
│                                                                                  │
│   Prompt Text ➔ Tokenizer (Byte-Level BPE, 8000 vocab) ➔ (1, 128) right-padded   │
│   ➔ Embedding (8000, 192)                                                        │
│   ➔ LSTM Layer 1 (256 units, return_sequences=True, dropout=0.15)                │
│   ➔ LSTM Layer 2 (256 units, return_sequences=True, dropout=0.15)                │
│   ➔ Dense Linear Logits (8000 units)                                             │
│   ➔ Logits Slicing at last_valid_position                                        │
│   ➔ Sampling Engine (Temperature, Top-K, Top-P, Logit Repetition Penalty)        │
│   ➔ BAN PAD(0), UNK(1), BOS(2); Stop on EOS(3)                                  │
│   ➔ Output Continuation (decoded via tokenizer_v3)                               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Hardware & Software Environment

- **Processor:** Intel® Core™ Ultra 5 125H (14 cores, 18 threads)
- **RAM:** 16 GB LPDDR5X
- **Operating System:** Windows 11 Pro 64-bit
- **Python:** 3.13.14 (in `.venv`)
- **Node.js & npm:** Node v24.11.1 / npm 11.6.2
- **Core Libraries:**
  - `tensorflow == 2.21.0`
  - `keras == 3.15.1`
  - `tokenizers == 0.23.3`
  - `numpy == 2.5.3`
  - `fastapi == 0.142.2`
  - `uvicorn == 0.54.0`
  - `react == 18.3.1`
  - `vite == 6.2.0`
  - `typescript == 5.7.3`

---

## 📊 Model Evaluation & Metrics

> [!IMPORTANT]
> The metrics below are held-out subword next-token prediction metrics specific to each model's tokenizer. They do not represent the percentage of complete stories that are correct, nor is 90% accuracy a valid target for open-ended text generation across an 8,000-token vocabulary.

| Metric | Story LSTM v3 (Active Default) | Story LSTM v2 (Preserved Baseline) | Description |
| :--- | :---: | :---: | :--- |
| **Model Version** | `story-lstm-v3-colab` | `story_lstm_v2` | Training candidate |
| **Tokenizer** | Byte-Level BPE (8,000 tokens) | Byte-Level BPE (8,000 tokens) | Subword tokenization |
| **Trainable Parameters** | **4,577,088** | 4,577,088 | 2-layer stacked LSTM |
| **Context Length (seq_len)**| **128** | 128 | Autoregressive context window |
| **Test Loss** | **2.4072 nats / subword** | 4.1780 nats / subword | Cross-entropy loss |
| **Test Perplexity** | **11.10** | 65.62 | Evaluated on matched tokenizer |
| **Next-Token Accuracy** | **46.41%** | 27.47% | Correct next token on held-out split |
| **Average Generation Latency**| **~1,168 ms** | ~1,005 ms | CPU inference for 50 tokens |

---

## 🔬 Empirical v2 vs v3 Side-by-Side Comparison

A 30-prompt evaluation across 8 narrative categories was conducted using identical generation parameters (`T=0.75, Top-K=40, Top-P=0.90, Repetition Penalty=1.15, Seed=42`).

- **Artifacts Generated:**
  - `artifacts/reports/v2_vs_v3_generation_comparison.json` (complete outputs and diagnostics)
  - `artifacts/reports/v2_vs_v3_generation_comparison.md` (readable side-by-side markdown report)
  - `artifacts/reports/human_evaluation_sheet_v3.csv` (standardized evaluation sheet)
- **Findings:**
  - **Syntax & Grammar:** v3 consistently forms complete English clauses with proper punctuation, avoiding v2's broken quote fragments (`"But, Lily's great..."`).
  - **Narrative Continuity:** v3 maintains prompt entities and dialogue flow (e.g. continuing dragon/knight stories in prompt 2 rather than abruptly introducing unrelated characters).
  - **Repetition Elimination:** v3 eliminated the repetitive degenerate template loops (`"We are my car. But the best is your friends," Ben asked.`) and premature 3-word terminations (`"of the ground."`, `"get the door."`) that afflicted v2.

---

## 🚀 Quickstart & Running the Application

### 1. Run the Single-Server Application
Builds the React frontend and serves both API and Web UI on port 8000:
```powershell
.\scripts\run_all.ps1
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!

### 2. Alternative: Running in Development Mode
- **Backend:**
  ```powershell
  .venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
  ```
- **Frontend (Vite dev server with proxy):**
  ```powershell
  cd frontend
  npm run dev
  ```
  Access frontend at `http://localhost:5173`.

---

## ⚙️ Model Configuration & Rollback Procedure

### Active Model Selection
By default, the application runs **Story LSTM v3**. To select a specific version at launch, set `ACTIVE_MODEL_VERSION`:

- **Run with Story LSTM v3 (Default):**
  ```powershell
  $env:ACTIVE_MODEL_VERSION="v3"
  .venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
  ```

- **Instant Rollback to Story LSTM v2:**
  ```powershell
  $env:ACTIVE_MODEL_VERSION="v2"
  .venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
  ```

### Verifying Active Model Version
Check active model status via API:
```bash
curl http://127.0.0.1:8000/api/v1/model/status
```
Example response:
```json
{
  "status": "ready",
  "model_name": "story_continuation_lstm",
  "version": "3.0.0",
  "vocab_size": 8000,
  "device": "CPU",
  "parameters_count": 4577088,
  "tokenizer_type": "subword_bytelevel_bpe_v3",
  "active_version": "v3",
  "available_versions": ["v2", "v3"]
}
```

---

## 🧪 Testing & Verification

Run the full pytest suite (68 tests across ML, backend API, tokenizer, shapes, and v3 integration):
```powershell
.\.venv\Scripts\pytest.exe
```
Result: **68 passed** in ~18 seconds.

Build the frontend bundle:
```powershell
npm --prefix frontend run build
```
Result: **Zero errors**, compiled in under 2 seconds.

---

## 📂 Artifact Placement & Provenance

All model weights, tokenizers, metadata, and reports are located in `artifacts/`:
- `artifacts/models/story_lstm_v3.keras` (SHA-256: `a218be909eab68d1084e63a9fce60999119423d0146dccfe4e4a07f7094657e8`)
- `artifacts/tokenizers/tokenizer_v3.json` (SHA-256: `cec16b4905fa068b6294225066f80127bd97769ab5a898cf2108fffd9a7f0843`)
- `artifacts/metadata/model_v3.json` (Full model specifications, hyperparameters, and hashes)
- `artifacts/metadata/tokenizer_v3_metadata.json` (BPE metadata, special tokens)
- `artifacts/metadata/dataset_manifest_v3.json` (TinyStories source and streaming manifest)
- `artifacts/metadata/split_manifest_v3.json` (90/5/5 document split policy)
- `artifacts/metrics/test_metrics_v3.json` (Held-out evaluation metrics)
- `artifacts/metrics/training_history_v3.json` (20-epoch training loss and accuracy logs)
- `artifacts/reports/training_report_v3.md` (Colab training report)
- `artifacts/reports/v2_vs_v3_generation_comparison.json` (30-prompt side-by-side evaluation)
- `artifacts/reports/human_evaluation_sheet_v3.csv` (Blank human assessment rubrics)
- `artifacts/models/story_lstm_v2.keras` & `artifacts/tokenizers/tokenizer_v2.json` (Preserved rollback baseline)
