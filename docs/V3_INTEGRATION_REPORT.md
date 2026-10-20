# Story LSTM v3 Integration Report

## 1. Executive Summary
Story LSTM v3 (`story-lstm-v3-colab`), trained on 50,000 TinyStories documents with an 8,000 Byte-Level BPE subword vocabulary and a 2-layer stacked LSTM architecture, has been safely and fully integrated into the local story-generation application (`e:\mdm`). 

The baseline Story LSTM v2 model and tokenizer have been preserved intact for immediate rollback. Compatibility checks, hash verifications, inference logic adjustments (right-padded input and last-valid-token logit sampling), unit/regression testing (68 passed tests), frontend compilation, end-to-end service testing, and an empirical 30-prompt comparative evaluation were all executed.

Based on empirical generation evaluation, **Story LSTM v3 has been activated as the default model**. It demonstrates significantly superior sentence coherence and dialogue handling while eliminating the repetitive degenerate loops and 3-word dead-end terminations that afflicted v2.

---

## 2. Artifact Hash Verification Results

The bundle was extracted to a temporary staging folder, where file presence and integrity were validated against `model_v3.json` before copying to production paths.

| Artifact | Production Path | Expected SHA-256 (from `model_v3.json`) | Computed SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **Model** | `artifacts/models/story_lstm_v3.keras` | `a218be909eab68d1084e63a9fce60999119423d0146dccfe4e4a07f7094657e8` | `a218be909eab68d1084e63a9fce60999119423d0146dccfe4e4a07f7094657e8` | **MATCH (PASS)** |
| **Tokenizer** | `artifacts/tokenizers/tokenizer_v3.json` | `cec16b4905fa068b6294225066f80127bd97769ab5a898cf2108fffd9a7f0843` | `cec16b4905fa068b6294225066f80127bd97769ab5a898cf2108fffd9a7f0843` | **MATCH (PASS)** |

Additional bundle metadata and evaluation files placed:
- `artifacts/metadata/model_v3.json`
- `artifacts/metadata/tokenizer_v3_metadata.json`
- `artifacts/metadata/dataset_manifest_v3.json`
- `artifacts/metadata/split_manifest_v3.json`
- `artifacts/metrics/test_metrics_v3.json`
- `artifacts/metrics/training_history_v3.json`
- `artifacts/samples/sample_generations_v3.json`
- `artifacts/reports/training_report_v3.md`

---

## 3. Runtime Environment & Isolated Model Load Verification

- **Operating System:** Windows 11 Pro 64-bit
- **Python Version:** 3.13.14 (tags/v3.13.14:fd17997)
- **TensorFlow Version:** 2.21.0
- **Keras Version:** 3.15.1
- **Hugging Face Tokenizers:** 0.23.3
- **NumPy Version:** 2.5.3

### Model Architecture Verification
- Model loaded successfully with `compile=False` using native Keras 3.15.1.
- Total Trainable Parameters: **4,577,088**
- Layers:
  1. `token_input`: `(None, 128)`
  2. `token_embedding`: `(None, 128, 192)`
  3. `not_equal`: `(None, 128)` (Keras boolean mask where `token_input != 0`)
  4. `lstm_1`: `(None, 128, 256)`
  5. `dropout_1`: `(None, 128, 256)`
  6. `lstm_2`: `(None, 128, 256)`
  7. `dropout_2`: `(None, 128, 256)`
  8. `token_logits`: `(None, 128, 8000)` (Dense linear layer, raw logits)
- Forward pass output shape confirmed: `(1, 128, 8000)`.
- Graph execution via `tf.function(..., reduce_retracing=True)` confirmed operational for sub-millisecond CPU dispatch.

---

## 4. Key Implementation Details & Inference Adaptations

1. **Strict Model-Tokenizer Pairing**:
   - `model_service.py` enforces that v3 model weights are only loaded and paired with `tokenizer_v3.json`.
   - Hashes and vocabulary signatures are validated at startup. If v3 is missing or corrupt, `ModelService` falls back to v2.
2. **Right-Padding Alignment & Last-Valid Position**:
   - v3 training pipeline placed input tokens left-aligned with right-padding (`PAD_ID=0`).
   - Inference creates `arr = np.full((1, 128), 0, dtype=np.int32)` and populates `arr[0, :len(window)] = window`.
   - The logits slice is taken precisely at `pred[0, len(window) - 1, :]`. This ensures sampling never takes place at a PAD token position.
3. **Special Token Handling**:
   - Special tokens: `PAD=0`, `UNK=1`, `BOS=2`, `EOS=3`.
   - Empty prompt receives `[BOS_ID=2]` as initial context.
   - During sampling, `PAD`, `BOS`, and `UNK` logits are masked to `-np.inf`.
   - Generation terminates immediately upon emitting `EOS_ID=3`.
   - Repetition penalties skip banned special tokens.
4. **Logits-Space Sampling**:
   - `sample_next_token` operates in float64 logit space: bans special tokens, applies repetition penalty, scales by temperature, applies Top-K filtering, performs numerically stable softmax, and computes Top-P nucleus mass cutoff.
5. **Rollback & Version Registry**:
   - `backend/app/model_service.py` maintains an internal multi-model registry holding both `v2` and `v3` instances simultaneously in memory (<40MB total footprint).
   - Configuration via `ACTIVE_MODEL_VERSION` environment variable or API request field `model_version` allows dynamic switching and side-by-side evaluation.

---

## 5. Automated Test Results

The full test suite was executed across the repository:
```powershell
.\.venv\Scripts\pytest.exe
```

### Result: 68 Passed, 0 Failed (18.22s)

Summary of test suites:
- `backend/tests/test_api.py`: 8 tests passed
  - Health probe, model status, model metrics, input validation (empty prompt, zero temp, token overflow), generation, side-by-side compare.
- `ml/tests/test_ml.py`: 5 tests passed
  - Tokenizer encode/decode, unknown token handling, sequence creation, model output shapes, bounded next-token sampling.
- `ml/tests/test_v2_pipeline.py`: 8 tests passed
  - Tokenizer v2 special token mappings, roundtrip fidelity, zero-UNK rate, masked loss ignoring PAD, masked accuracy.
- `tests/test_artifact_loading.py`: 1 test passed
- `tests/test_config.py`: 3 tests passed
- `tests/test_generation.py`: 4 tests passed
- `tests/test_metrics.py`: 4 tests passed
- `tests/test_model_shape.py`: 3 tests passed
- `tests/test_preprocessing.py`: 4 tests passed
- `tests/test_sequence_creation.py`: 3 tests passed
- `tests/test_smoke.py`: 1 test passed (synthetic mini-corpus train, reload, and generation)
- `tests/test_tokenizer.py`: 4 tests passed
- `tests/test_v3_integration.py`: 20 tests passed
  - SHA-256 verification and artifact presence.
  - v3 tokenizer loading, vocab size 8000, special tokens (0, 1, 2, 3), roundtrip fidelity.
  - Input/output shapes `(1, 128, 8000)`, finite logits validation.
  - 1-token prompt, normal prompt, exact 128-token prompt, >128-token prompt handling.
  - Special token banning and EOS termination.
  - Repetition penalty suppression and greedy sampling.
  - Model service registry coexistence, runtime version switching (v3 -> v2 -> v3).
  - Corrupt v3 detection and rollback to v2.
  - API v3 override and v2 regression compatibility.

---

## 6. Frontend Build & End-to-End Smoke Test

### Frontend Build
Command:
```powershell
npm --prefix frontend run build
```
Output:
```
> story-continuation-frontend@2.0.0 build
> tsc && vite build

vite v6.4.3 building for production...
transforming...
✓ 1598 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.91 kB │ gzip:  0.53 kB
dist/assets/index-D2vvMKg5.css    4.96 kB │ gzip:  1.64 kB
dist/assets/index-DU7Q8Ip5.js   182.78 kB │ gzip: 56.15 kB
✓ built in 1.44s
```
**Status: Zero errors.**

### End-to-End Local Smoke Test
A live test against the FastAPI app serving the compiled static frontend yielded:
- `GET /api/v1/health`: `200 OK` (status: healthy, model_loaded: true, version: 3.0.0)
- `GET /api/v1/model/status`: `200 OK` (model: story_continuation_lstm, version: 3.0.0, active_version: v3, available: ['v2', 'v3'])
- `POST /api/v1/generate`: `200 OK` (returned valid story continuation in 552 ms)
- `GET /`: `200 OK` (text/html; charset=utf-8, index.html successfully served)

---

## 7. Fixed 30-Prompt Comparative Evaluation Findings

Evaluation script `scripts/evaluate_v2_vs_v3.py` executed 30 prompts from `artifacts/reports/evaluation_prompts_suite.json` under identical sampling conditions (`max_new_tokens=50, T=0.75, Top-K=40, Top-P=0.90, Repetition Penalty=1.15, Seed=42`).

### Quantitative Summary
| Metric | Story LSTM v2 (Baseline) | Story LSTM v3 (Colab Candidate) | Note |
| :--- | :---: | :---: | :--- |
| **Mean Latency** | 1,005.1 ms | 1,168.4 ms | +163.3 ms difference (16%) |
| **Distinct-1 (Unigram)** | 0.8917 | 0.8063 | v3 produces more coherent grammatical sentences with standard English function words |
| **Distinct-2 (Bigram)** | 0.9912 | 0.9759 | High bigram diversity maintained |
| **Repetition-4 Rate** | 0.0000 | 0.0000 | No degenerate 4-gram loops under 1.15 repetition penalty |

### Qualitative Review of Outputs
1. **Elimination of Degenerate Loops**:
   - In v2, several prompts (e.g. Prompt 8, 11, 20) fell into an identical degenerate loop template:
     `"she had fun. But they said, 'We are my car. But the best is your friends,' Ben asked. He was so happy and gave it to be very very too. She wanted to make a small room."`
   - In v3, this failure mode was 100% eliminated. Prompt 20 ("Greedy monkey Bob...") produced:
     `"it was important to be honest and respectful. He learned that sometimes things can be different, like that you have done something special."`
2. **Elimination of Abrupt 3-Word Dead Ends**:
   - In v2, 7 out of 30 prompts terminated prematurely with nonsensical 3-word fragments like `"of the ground."`, `"the old room."`, or `"get the door."` (e.g., Prompt 2, 10, 12, 17, 19, 21, 27).
   - In v3, all 30 prompts generated full narrative arcs with proper sentence closures or natural story endings (`"The End."`, `"The end."`, `"And they all lived happily ever after."`).
3. **Dialogue & Entity Consistency**:
   - In v2, dialogue prompts (Prompts 13-16) frequently generated malformed quotes with missing tags or random names (`"But, Lily's nice. I can eat you are you.I will have her friend..."`).
   - In v3, dialogues preserved conversation partners and natural responses (`"Yes, I want to see what it is for!" Mia smiled and said, "That's right, Jake. Let's go and have some fun together."`).

---

## 8. Default Activation Decision

**Decision:** **Story LSTM v3 is activated as the default active model.**

**Justification:**
1. **Empirical Quality**: Side-by-side evaluation demonstrates that v3 produces vastly superior narrative coherence, complete sentences, and contextually appropriate dialogue while completely eliminating v2's repetitive loop templates and dead-end phrases.
2. **Zero Regressions**: All 68 backend, ML, shape, and API tests pass without failure.
3. **Instant Rollback Preserved**: v2 weights, tokenizer, and pipelines remain untouched in `artifacts/models/story_lstm_v2.keras` and `artifacts/tokenizers/tokenizer_v2.json`.

---

## 9. Rollback Instructions

If an operational issue arises or v2 output is desired:

1. **Environment Variable Rollback**:
   Set `ACTIVE_MODEL_VERSION="v2"` in your shell or `.env`:
   ```powershell
   $env:ACTIVE_MODEL_VERSION="v2"
   .venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   ```
2. **Runtime API Switch**:
   Pass `"model_version": "v2"` in the JSON payload to `/api/v1/generate`.
3. **Web UI Toggle**:
   Click the **"v2 Baseline"** button in the Sampling Hyperparameters panel on the Generator page.

---

## 10. Limitations & Unexecuted Checks

1. **Normalized Cross-Tokenizer Perplexity**:
   - The reported v3 perplexity (11.10) and next-token accuracy (46.41%) are calculated on the v3 subword tokenizer.
   - Direct comparison with v2 word-level perplexity without identical character-level bits-per-byte normalization is mathematically invalid. We document the raw reported metrics without claiming an unsubstantiated percentage improvement.
2. **GPU Acceleration**:
   - Native Windows TensorFlow >= 2.11 does not support native CUDA. The application is tuned and tested CPU-first with oneDNN and `tf.function` graph compilation, running in ~1.1 seconds per 50-token sequence.

---

## 11. Final Verification Checklist

- [x] Model and tokenizer SHA-256 values match metadata.
- [x] v3 model and v3 tokenizer are always loaded together.
- [x] v2 model and tokenizer remain intact for rollback.
- [x] Keras/TensorFlow compatibility is tested in the actual project environment.
- [x] Inference samples from raw logits at the last valid prompt position.
- [x] PAD/BOS/EOS/UNK handling is correct and tested.
- [x] Temperature, Top-K, Top-P, and repetition penalty tests pass.
- [x] Short, normal, and long prompt tests pass.
- [x] Existing ML and API tests pass.
- [x] npm run build completes inside the frontend directory with zero errors.
- [x] Local frontend/backend end-to-end smoke test is performed where possible.
- [x] v2/v3 generation comparison is saved and not cherry-picked.
- [x] Metrics are described with the correct tokenizer and interpretation.
- [x] Documentation records actual test/build results and rollback steps.
- [x] Existing model behavior is not broken by the upgrade.

### Deployment Checklist
- [x] npm run build → zero errors
- [x] Deploy to Vercel → test fixes on live URL *(N/A: Local single-server FastAPI/Uvicorn architecture)*
- [x] Check Vercel Function Logs for API errors *(N/A: Local single-server deployment)*
