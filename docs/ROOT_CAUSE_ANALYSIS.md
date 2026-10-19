# Root Cause Analysis: Story Continuation Generator Using Stacked LSTM

---

## Executive Summary
An exhaustive inspection of the repository (`ml/src/`, `src/`, `backend/app/`, `artifacts/`, `configs/`, `scripts/`, and `tests/`) was conducted to diagnose why the existing generator produces broken clauses, grammatical errors, repetition loops, and poor narrative continuation.

The inspection revealed **six fundamental architectural defects** spanning the training objective, tokenizer, data scale, loss calculation, sampling mathematics, and module organization.

---

## 1. Inventory of Active vs. Legacy Modules

| Path | Status | Verification & Evidence |
| :--- | :--- | :--- |
| `ml/src/` | **ACTIVE** | Directly imported by `backend/app/model_service.py`, `backend/app/generator.py`, `scripts/train.ps1`, `scripts/evaluate.ps1`. |
| `backend/app/` | **ACTIVE** | Primary FastAPI service powering `/api/v1/generate`, `/health`, and serving compiled React `frontend/dist`. |
| `src/` | **LEGACY** | Unused historical remnant from earlier Streamlit prototype. |
| `app/` | **LEGACY** | Old Streamlit application directory (not imported by active FastAPI backend). |
| `configs/config.yaml` | **LEGACY** | Superseded by `ml/configs/base.yaml`. |
| `ml/configs/base.yaml` | **ACTIVE** | Loaded by `ml/src/utils.py` and active training scripts. |

---

## 2. Baseline Architecture & Performance Metrics

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Model Type** | 2-Layer Word-Level LSTM | `Embedding -> LSTM(256, seq=True) -> LSTM(256, seq=False) -> Dense(5044, softmax)` |
| **Total Parameters** | `2,861,492` | $645,632$ (Embedding) + $394,240$ (LSTM 1) + $525,312$ (LSTM 2) + $1,296,308$ (Dense) |
| **Vocabulary Size** | `5,044` entries | Word-level, lowercased, special tokens `<PAD>`(0), `<UNK>`(1), `<START>`(2), `<END>`(3) |
| **Context Length** | `50` tokens | Fixed sliding window |
| **Training Corpus** | `129,480` tokens | 62 public-domain Grimm & Andersen fairy tales |
| **Train / Val / Test Loss** | `3.7991` / `5.0532` / `5.2177` | Sparse Categorical Cross-Entropy |
| **Test Perplexity (PPL)** | `184.51` | $\exp(5.2177)$ |
| **Test Next-Token Accuracy** | `19.28%` | Top-1 accuracy on held-out test tales |
| **Inference Latency** | `~18 - 25 ms` / token | Multi-threaded CPU on Intel Core Ultra 5 125H |

---

## 3. Confirmed Root Causes & Corrective Actions

### Defect 1: Inefficient Many-to-One Training Objective (CRITICAL)
* **Evidence:** In `ml/src/model.py`, the second LSTM layer uses `return_sequences=False`. In `ml/src/build_sequences.py`, sliding windows of length 50 map to a single scalar target $y = \text{tokens}[i + 50]$.
* **Impact:** For a 50-step sequence, 49 intermediate forward states receive **zero loss supervision**. The model discards internal representations across the sequence, forcing a 50x redundant memory expansion in `data/processed/train_x.npy` while learning only one token per step.
* **Corrective Action (Phase 4):** Convert to a **many-to-many autoregressive language modeling objective**:
  $$X = [w_1, w_2, \dots, w_{L-1}], \quad Y = [w_2, w_3, \dots, w_L]$$
  Both LSTM layers set `return_sequences=True`. A `TimeDistributed` Dense projection produces next-token predictions at every timestep $t \in [1, L-1]$, supervised simultaneously with masked cross-entropy.

---

### Defect 2: Rigid Word-Level Tokenizer & High Unknown-Token Rate (CRITICAL)
* **Evidence:** `ml/src/tokenizer.py` relies on `re.sub` whitespace splitting and lowercasing, restricted to 5,044 words.
* **Impact:** 
  1. Any morphological variation, capitalization, proper noun, or unseen vocabulary item is mapped to `<UNK>` (index 1).
  2. Loss of true case formatting (everything becomes lowercased).
  3. Zero ability to represent subword compositionality (e.g., *kingly*, *unlocked*, *woodcutter*).
* **Corrective Action (Phase 3):** Implement a **SentencePiece / Byte-Pair Encoding (BPE)** subword tokenizer with an ~8,000 token vocabulary trained exclusively on the training split. Preserves case, punctuation, numbers, and decomposes unseen words into known subwords with **0% OOV rate**.

---

### Defect 3: Severe Data Sparsity & Micro-Corpus (CRITICAL)
* **Evidence:** The current training set consists of only 62 stories (~129,480 tokens) from 19th-century fairy tales.
* **Impact:** The model lacks exposure to diverse syntactic sentence structures, modern grammatical dialogue, varied predicates, and logical causal connectors. Overfitting occurs rapidly after 4–5 epochs ($val\_loss$ plateaus at 5.05 while training loss drops to 3.79).
* **Corrective Action (Phase 2):** Incorporate the canonical **`roneneldan/TinyStories`** dataset from Hugging Face in streaming mode with buffered shuffling, mixed with the fairy-tale corpus at a target token ratio of **85% TinyStories / 15% Classic Fairy Tales** targeting ~2–5 million tokens within the laptop CPU budget.

---

### Defect 4: Flawed Sampling Mathematics in Generation Engine (HIGH)
* **Evidence:** In `backend/app/generator.py`:
  1. *Repetition Penalty:* Applied directly to normalized probabilities (`probs[t_id] /= repetition_penalty`), which breaks the normalization sum and distorts relative entropy.
  2. *Temperature Scaling:* The model outputs softmax probabilities, so the code inverts them with `np.log(probs) / T`, losing precision for low-probability candidates before exponentiating again.
  3. *Top-P (Nucleus) Logic:* Executed after Top-K zeroed out entries, corrupting the cumulative sum array and leading to greedy single-token collapse when `cumulative[0] >= top_p`.
* **Corrective Action (Phase 6):**
  1. Model outputs raw logits or unnormalized scores.
  2. Apply logit-based repetition penalty:
     $$z_i \leftarrow \begin{cases} z_i / \text{penalty} & \text{if } z_i > 0 \\ z_i \times \text{penalty} & \text{if } z_i \le 0 \end{cases}$$
  3. Scale logits directly by temperature: $z_i \leftarrow z_i / T$.
  4. Implement robust Top-K filtering and correct descending-cumulative-mass Top-P thresholding with proper renormalization.

---

### Defect 5: Absence of Padding Masking in Loss Computation (MEDIUM)
* **Evidence:** In `ml/src/train.py`, loss is compiled as standard `SparseCategoricalCrossentropy`. Left-padded tokens in shorter documents are passed into the embedding layer without masking (`mask_zero=False`), causing the LSTM state to track padding tokens as active representations.
* **Impact:** In many-to-many training, supervising `<PAD>` targets would severely degrade fluency.
* **Corrective Action (Phase 4):** Implement explicit sample weighting / loss masking:
  $$\mathcal{L} = \frac{\sum_{t} \mathbb{I}(y_t \ne \text{PAD\_ID}) \cdot \text{CrossEntropy}(z_t, y_t)}{\sum_t \mathbb{I}(y_t \ne \text{PAD\_ID})}$$

---

### Defect 6: Evaluation Rigor & Prompt Diversity Gaps (MEDIUM)
* **Evidence:** Previous evaluations relied on 4 sample prompts, with Distinct-N cited as proof of quality. Distinct-N measures lexical uniqueness, not narrative coherence or grammar.
* **Impact:** Grammatically broken sentences like *"said the king what he will not be find him into the wood"* achieved high Distinct-2 scores despite failing basic English grammar.
* **Corrective Action (Phase 7):** Establish a standardized **30-prompt evaluation suite** spanning 8 narrative categories, recording masked test perplexity, token accuracy, Distinct-1/2, repetition ratios, CPU inference latency (median & P95), and a standardized 1-to-5 human evaluation rubric.

---

## 4. Architectural Target & Transition Plan

```
Current Baseline (v1):
[Input 50] -> Embedding(5044, 128) -> LSTM(256, seq=True) -> LSTM(256, seq=False) -> Dense(5044, softmax) -> Single Target

Upgraded Candidate (v2):
[Input L=128] -> Embedding(8000, 192) -> LSTM(256, seq=True) -> Dropout(0.2)
             -> LSTM(256, seq=True) -> Dropout(0.2) -> Dense(8000, from_logits=True)
             -> Timestep-shifted Targets [y_1 ... y_L] with Masked Loss
```
