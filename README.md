# 📖 Story Continuation Generator Using Stacked LSTM

> A production-quality, locally trained **2-Layer LSTM Neural Language Model** paired with a modern **FastAPI** backend and **React + TypeScript + Vite** frontend. Built CPU-first for Windows 11 on the **Intel Core Ultra 5 125H**. Zero external generative APIs.

---

## 🌟 Overview & Problem Statement

Language generation often relies on multi-billion parameter cloud LLMs that introduce network latency, vendor lock-in, and unpredictable costs. This project implements a **100% self-contained autoregressive language model** using deep recurrent neural networks:

- **Locally Trained:** 2-layer stacked LSTM trained from scratch on 129k tokens of classic fairy tales (Grimm & Andersen).
- **CPU-First Performance:** Optimized for thin-and-light laptop CPUs (Intel Core Ultra 5 125H) without requiring NVIDIA CUDA.
- **Modern Single-Server Stack:** FastAPI backend serving the compiled Vite React frontend on port `8000`.
- **Dynamic Sampling Engine:** Next-token generation featuring Temperature scaling, Top-K truncation, Top-P (Nucleus) sampling, and Repetition Penalties.

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │   React 18 + TypeScript + Vite (Port 8000)   │
                    │   - PromptEditor with Preset Chips           │
                    │   - Dynamic Temperature / Top-K Controls     │
                    │   - OutputCard with Performance Diagnostics  │
                    └──────────────────────┬───────────────────────┘
                                           │ HTTP POST /api/v1/generate
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │            FastAPI Backend Service           │
                    │   - Pydantic Request/Response Validation     │
                    │   - Thread-safe ModelService Singleton       │
                    │   - Health & Status Probes                   │
                    │   - Static SPA Catch-all Server              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             LSTM Inference Pipeline                              │
│                                                                                  │
│   Prompt Text ➔ Tokenizer (word2idx) ➔ Context Window Tensor (1, 50)             │
│   ➔ Embedding (5004, 128)                                                        │
│   ➔ LSTM Layer 1 (256 units, return_sequences=True, dropout=0.20)                │
│   ➔ LSTM Layer 2 (256 units, return_sequences=False, dropout=0.20)               │
│   ➔ Dense Softmax (5004 units)                                                   │
│   ➔ Sampling Engine (Temperature, Top-K, Top-P, Repetition Penalty)              │
│   ➔ Output Continuation                                                          │
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
  - `fastapi == 0.142.2`
  - `uvicorn == 0.54.0`
  - `react == 18.3.1`
  - `vite == 6.2.0`
  - `typescript == 5.7.3`

---

## 📊 Evaluation & Model Performance

| Metric | Recorded Value | Description |
| :--- | :--- | :--- |
| **Best Validation Loss** | **5.0532** | Categorical cross-entropy over 5,004 vocab items |
| **Model Perplexity** | **156.52** | Effective word uncertainty branching factor |
| **Next-Token Accuracy** | **23.14%** | Accuracy on held-out test tales |
| **Latency per Token** | **~20 ms / token** | CPU inference on Intel Core Ultra 5 |
| **Total Parameters** | **2,453,804** | Trainable weights and recurrent state matrices |
| **Dataset Size** | **129,480 tokens** | 62 curated stories from Project Gutenberg |

---

## 🚀 Quickstart & Running the Application

### 1. Automated Setup
Run the setup PowerShell script:
```powershell
.\scripts\setup.ps1
```

### 2. Run the Single-Server Application
Builds the React frontend (if needed) and serves both API and Web UI on port 8000:
```powershell
.\scripts\run_all.ps1
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!

### 3. Alternative: Running in Development Mode
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

## 📡 API Specification

Interactive Swagger UI documentation is available at `http://127.0.0.1:8000/docs`.

### Key Endpoints:
- `GET /api/v1/health`: Readiness probe and model status.
- `GET /api/v1/model/status`: Parameter count, vocabulary size, and architecture specs.
- `GET /api/v1/model/metrics`: Quantitative training and evaluation loss/perplexity.
- `POST /api/v1/generate`: Autoregressive generation endpoint.

#### Example Request:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Once upon a time in a deep dark forest and",
       "max_new_tokens": 50,
       "temperature": 0.8,
       "top_k": 40,
       "top_p": 0.90
     }'
```

---

## 🧪 Testing & Verification

Run the full pytest suite for ML and API test clients:
```powershell
.venv\Scripts\pytest.exe ml/tests/ backend/tests/ -v
```
All 12 unit and integration tests pass cleanly.

---

## 📜 Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── generator.py
│   │   ├── health.py
│   │   ├── main.py
│   │   ├── model_service.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── static_server.py
│   ├── tests/
│   │   └── test_api.py
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── styles/
│   │   ├── types/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── ml/
│   ├── configs/
│   │   └── base.yaml
│   ├── src/
│   │   ├── build_sequences.py
│   │   ├── evaluate.py
│   │   ├── generate_cli.py
│   │   ├── model.py
│   │   ├── prepare_data.py
│   │   ├── tokenizer.py
│   │   ├── train.py
│   │   └── utils.py
│   └── tests/
│       └── test_ml.py
├── artifacts/
│   ├── metadata/
│   ├── metrics/
│   ├── models/
│   ├── plots/
│   └── tokenizers/
├── scripts/
│   ├── setup.ps1
│   ├── train.ps1
│   ├── evaluate.ps1
│   └── run_all.ps1
├── PROJECT_REPORT_NOTES.md
└── README.md
```
