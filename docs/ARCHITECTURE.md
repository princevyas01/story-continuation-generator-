# System Architecture Specification: Story Continuation Generator

**Project:** MDM Foundation of Generative AI - Story Continuation Generator (LSTM)  
**Hardware Node:** ASUS Zenbook 14 OLED (Intel Core Ultra 5 125H, 14 Cores / 18 Threads, 16 GB LPDDR5X RAM)  
**Execution Strategy:** Local CPU-First Execution with oneDNN Vector Acceleration

---

## 1. End-to-End Architectural Topology

The system comprises four decoupled, auditable layers: Data Engineering, Recurrent Modeling & Training, Autoregressive Inference, and User Demonstration.

```mermaid
flowchart TD
    subgraph Data Pipeline
        RAW["data/raw/ (Gutenberg Texts)"] --> CLEAN["Text Normalizer (NFKC, Punctuation Space, Lowercase)"]
        CLEAN --> SPLIT["Document Partition (80% Train / 10% Val / 10% Test)"]
        SPLIT --> MANIFEST["data/manifests/dataset_manifest.json"]
        SPLIT --> TOK["StoryTokenizer (Word Vocab ~5,044)"]
        TOK --> VOCAB_ART["artifacts/tokenizer_vocab.json"]
        SPLIT --> WIN["Sliding Window Sequence Extraction (seq_len=50, step=2)"]
    end

    subgraph Training Engine
        WIN --> TF_IN["Input Pairs: (X, y) (Batch: 64)"]
        TF_IN --> EMB["Embedding Layer (dim=128)"]
        EMB --> LSTM1["LSTM Layer 1 (units=256, dropout=0.2, seq=True)"]
        LSTM1 --> LSTM2["LSTM Layer 2 (units=256, dropout=0.2, seq=False)"]
        LSTM2 --> SOFTMAX["Dense Projection Head (Vocab_Size, Softmax)"]
        SOFTMAX --> LOSS["Sparse Categorical Cross-Entropy"]
        LOSS --> OPT["Adam Optimizer (clipnorm=1.0, lr=0.002)"]
        OPT --> CBS["Callbacks: Checkpoint, EarlyStopping, ReduceLR"]
        CBS --> BEST_M["models/best/story_lstm.keras"]
        CBS --> REPORT["artifacts/reports/training_report.md"]
        CBS --> PLOTS["artifacts/plots/loss_curve.png"]
    end

    subgraph Inference & Evaluation
        BEST_M --> GEN["Autoregressive Sampling Loop (src/generate.py)"]
        VOCAB_ART --> GEN
        GEN --> CONTROLS["Sampling Controls: Temp, Top-k, Top-p, Repetition Penalty"]
        CONTROLS --> EVAL["Evaluation Suite (src/evaluate.py)"]
        EVAL --> BASELINE["Bigram Baseline Comparison"]
        EVAL --> SAMPLES["artifacts/samples/sample_generations.json"]
        EVAL --> HUMAN_WS["artifacts/reports/human_evaluation_sheet.csv"]
    end

    subgraph Demonstration UI
        BEST_M -.->|st.cache_resource| ST["Streamlit Web Application (app/app.py)"]
        VOCAB_ART -.->|st.cache_resource| ST
        PLOTS -.-> ST
        SAMPLES -.-> ST
        ST --> USER_IN["Interactive Prompt Input & Sliders"]
        USER_IN --> UI_OUT["Formatted Story Card + Lexical Metrics + Text Export"]
    end
```

---

## 2. Component Boundaries & Interfaces

| Component | Input | Output | Invariants & Failure Guards |
| :--- | :--- | :--- | :--- |
| **`src/preprocessing.py`** | Raw Gutenberg `.txt` files | Normalized stories & document splits | Strips legal preambles/postambles; enforces NFKC Unicode; document-level partition prevents data leakage. |
| **`src/tokenizer.py`** | Document token sequences | Word-to-index map & token IDs | Fixed indices for `<PAD>` (0), `<UNK>` (1), `<START>` (2), `<END>` (3); persists to deterministic JSON artifact. |
| **`src/sequences.py`** | Token ID streams | Windowed $(X, y)$ ndarrays | Input shape $(N, 50)$; sparse integer target $(N,)$; left-padding for short sequences; zero-copy batching. |
| **`src/model.py`** | Sequence tensor $(B, 50)$ | Logits / Softmax $(B, V)$ | 2 stacked LSTM layers with recurrent dropout; gradient clipping (`clipnorm=1.0`) prevents gradient explosion. |
| **`src/train.py`** | Config YAML & dataset | `.keras` checkpoints & metrics | Checkpointing saves only optimal validation loss model; early stopping terminates plateauing runs. |
| **`src/generate.py`** | User prompt string + params | Continuation string + stats | Left-pads short prompts; masks `<PAD>`/`<START>` tokens; applies temperature, top-$k$, top-$p$, and repetition penalty. |
| **`src/evaluate.py`** | Trained model & test set | Objective metrics & sweep table | Calculates test loss, Perplexity ($\exp(\mathcal{L})$), Distinct-1/2, repetition ratio, and bigram baseline comparison. |
| **`app/app.py`** | UI controls & user inputs | Interactive Streamlit demo | Uses `st.cache_resource` to avoid model reload across interactions; provides safe export without external dependencies. |

---

## 3. Data Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Student
    participant App as Streamlit (app.py)
    participant Engine as Generator (generate.py)
    participant Model as LSTM Model (.keras)
    participant Tokenizer as StoryTokenizer

    User->>App: Submits Prompt ("once upon a time...") & Hyperparameters (T=0.8, k=40)
    App->>Engine: generate_continuation(prompt, seq_len=50, max_tokens=50, T, k, p)
    Engine->>Tokenizer: encode(normalized_prompt)
    Tokenizer-->>Engine: token_ids [28, 14, 5, 89]
    Engine->>Engine: Left-pad to seq_len (shape: 1, 50)
    
    loop Autoregressive Generation Step (1 to max_tokens)
        Engine->>Model: Forward Pass (input_window)
        Model-->>Engine: Softmax probabilities over vocabulary (dim: 5044)
        Engine->>Engine: Mask <PAD>/<START>, apply Repetition Penalty, Scale by Temperature T
        Engine->>Engine: Filter Top-k and Top-p candidates, Sample Token from distribution
        Engine->>Engine: Append Token to Sequence Window (sliding 1 step)
        opt Reached <END> Token
            Engine->>Engine: Break Generation Loop
        end
    end

    Engine->>Tokenizer: decode(generated_token_ids)
    Tokenizer-->>Engine: continuation_text
    Engine-->>App: Result Payload (full_story, latency, token_count)
    App-->>User: Renders Story Card, Diversity Badges, and Download Button
```

---

## 4. Artifact Dependency Hierarchy

1. `configs/config.yaml` $\to$ Controls all parameters and file paths.
2. `scripts/prepare_data.py` $\to$ Generates `data/processed/*.txt`, `data/manifests/dataset_manifest.json`, and `artifacts/tokenizer_vocab.json`.
3. `src/train.py` $\to$ Consumes processed data and tokenizer; writes `models/best/story_lstm.keras`, `artifacts/plots/loss_curve.png`, `artifacts/metrics/metrics.json`, and `artifacts/reports/training_report.md`.
4. `src/evaluate.py` $\to$ Consumes test split and best model; writes `artifacts/samples/sample_generations.json`, `artifacts/reports/human_evaluation_sheet.csv`, and baseline benchmarks.
5. `app/app.py` $\to$ Loads trained model, tokenizer, and metrics artifacts to serve the demonstration UI.
