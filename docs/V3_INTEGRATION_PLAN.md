# Story LSTM v3 Integration Plan

## 1. Executive Summary & Objectives
This plan outlines the safe, verified integration of the newly trained Google Colab Story LSTM v3 candidate model into the existing local story-generation application (`e:\mdm`). 

Key integration objectives:
1. **Preserve Rollback**: Keep Story LSTM v2 completely intact and accessible as an immediate rollback target.
2. **Strict Pairing**: Ensure `story_lstm_v3.keras` and `tokenizer_v3.json` are loaded together as an inseparable pair; prevent accidental pairing with v2 tokenizer.
3. **Logits-based Inference**: Adapt generation to v3's raw logits output shape `(batch, 128, 8000)`, selecting the slice at the last valid prompt token position (`last_valid_position`), right-padding unused timesteps with `PAD_ID=0`.
4. **Sampling & Safety**: Ensure temperature, top-k, top-p nucleus sampling, and logit-space repetition penalty operate numerically stably without penalizing or emitting special tokens (`<PAD>`, `<BOS>`), stopping on `<EOS>`.
5. **Comprehensive Verification**: Validate SHA-256 hashes, execute automated ML and API tests, verify frontend build, run a 30-prompt side-by-side comparison, and establish a defensible default activation decision.

---

## 2. Confirmed Active System Paths & Baseline Architecture

### Model & Artifact Paths
- **Current v2 Model**: `artifacts/models/story_lstm_v2.keras` (Backup: `models/best/story_lstm_v2.keras`)
- **Current v2 Tokenizer**: `artifacts/tokenizers/tokenizer_v2.json`
- **Current v2 Metrics & Reports**: `artifacts/reports/evaluation_comparison_v2.json`, `artifacts/reports/training_report_v2.md`
- **Legacy v1 Model**: `artifacts/models/story_lstm.keras`
- **Legacy v1 Tokenizer**: `artifacts/tokenizers/tokenizer.json`

### Code Architecture & Active Components
- **Configuration**: `backend/app/config.py` (`Settings` class defining model/tokenizer paths and defaults)
- **Model Service**: `backend/app/model_service.py` (`ModelService` singleton managing loading, inference locks, and health status)
- **Generation Engine**: `backend/app/generator.py` (`sample_next_token` and `run_generation` implementing autoregressive loops and sampling)
- **Tokenizer Classes**: `ml/src/subword_tokenizer.py` (`SubwordStoryTokenizer` wrapping Hugging Face `tokenizers.Tokenizer`) and `ml/src/tokenizer.py` (`StoryTokenizer` word-level)
- **API Routing & Schemas**: `backend/app/routes.py` and `backend/app/schemas.py` (`/api/v1/generate`, `/api/v1/generate/compare`, `/api/v1/model/status`, `/api/v1/model/metrics`, `/api/v1/health`)
- **Frontend**: `frontend/src/` (React 18 + TypeScript + Vite; `App.tsx`, `pages/GeneratorPage.tsx`, `pages/MetricsPage.tsx`, `components/GenerationControls.tsx`, `api/client.ts`)

### Environment & Toolchain
- **Python**: 3.13.14 (64-bit on Windows)
- **TensorFlow**: 2.21.0
- **Keras**: 3.15.1
- **Tokenizers**: 0.23.3
- **NumPy**: 2.5.3
- **Frontend Build**: Vite v6.2.0 (`npm --prefix frontend run build` verified: zero errors)
- **Test Runner**: Pytest 9.1.1 (`.\.venv\Scripts\pytest.exe` verified: 48 passed)

---

## 3. Compatibility Risks & Mitigations

| Risk | Severity | Mitigation Strategy |
| :--- | :--- | :--- |
| **Logits vs Softmax Output** | High | v3 Dense head produces raw linear logits `(batch, 128, 8000)`, not probabilities. Ensure generation sets `from_logits=True`, performing numerically stable softmax within `sample_next_token`. |
| **Padding Alignment & Last Valid Position** | High | Colab v3 training pipeline placed input tokens left-aligned with right-padding of zeros (`PAD_ID=0`). Inference window must place tokens at `[0, :len(window)]` and sample specifically from `output[0, len(window) - 1, :]`. Sampling from position 127 would sample from PAD positions and produce garbage. |
| **Tokenizer Mismatch** | Critical | Pairing v3 weights with v2 tokenizer would scramble token embeddings. Enforce strict pair verification in `ModelService`, validating matching hashes or version signatures at load time. |
| **Model Serialization / Custom Loss** | Medium | The `.keras` file was trained with custom masked loss. Loading for inference must use `compile=False` to prevent Keras deserialization errors on non-compiled graphs. |
| **Special Token Leakage** | Medium | Banned token set `{PAD_ID=0, BOS_ID=2, UNK_ID=1}` must have logits masked to `-np.inf` during generation. Generation must stop immediately when `EOS_ID=3` is emitted. |
| **Repetition Penalty on Special Tokens** | Low | Repetition penalty must only modify valid non-special vocabulary indices present in recent history. |
| **Rollback Safety** | Critical | Do not overwrite v2 files. Implement version configuration (`MODEL_VERSION` env var or configuration flag with `"v2"` or `"v3"`). If v3 loading fails, automatically fall back cleanly to v2. |

---

## 4. Implementation Sequence

### Phase 1: Staging & Artifact Verification
1. Confirm all bundle files in `staging_v3/story_lstm_v3_colab_bundle/`.
2. Compute SHA-256 for `story_lstm_v3.keras` and `tokenizer_v3.json`.
3. Compare against expected values in `model_v3.json`:
   - Model expected: `a218be909eab68d1084e63a9fce60999119423d0146dccfe4e4a07f7094657e8`
   - Tokenizer expected: `cec16b4905fa068b6294225066f80127bd97769ab5a898cf2108fffd9a7f0843`
4. Copy artifacts to versioned locations:
   - `artifacts/models/story_lstm_v3.keras`
   - `artifacts/tokenizers/tokenizer_v3.json`
   - `artifacts/metadata/model_v3.json`
   - `artifacts/metadata/tokenizer_v3_metadata.json`
   - `artifacts/metadata/dataset_manifest_v3.json`
   - `artifacts/metadata/split_manifest_v3.json`
   - `artifacts/metrics/test_metrics_v3.json`
   - `artifacts/metrics/training_history_v3.json`
   - `artifacts/samples/sample_generations_v3.json`
   - `artifacts/reports/training_report_v3.md`

### Phase 2: Configuration & Service Layer Enhancement
1. Update `backend/app/config.py`:
   - Add `MODEL_V3_PATH`, `TOKENIZER_V3_PATH`, `METRICS_V3_PATH`, `MODEL_V3_METADATA_PATH`.
   - Add `ACTIVE_MODEL_VERSION` setting (defaulting initially to `"v2"`, switchable to `"v3"`).
2. Enhance `backend/app/model_service.py`:
   - Implement version-aware artifact resolution supporting `"v3"`, `"v2"`, and legacy `"v1"`.
   - Add pairing validation ensuring model and tokenizer versions match.
   - Load `story_lstm_v3.keras` with `compile=False`.
   - Warm up inference with a test forward pass `(1, 128)`.
   - Update `get_status()` to report active model version (`"3.0.0"` or `"2.0.0"`), vocabulary size, parameter count, and tokenizer type.
   - Expose fallback to v2 if v3 loading fails.

### Phase 3: Generation Engine Adaptation (v3 Semantics)
1. In `backend/app/generator.py`:
   - Support right-padded sequence windows for v3:
     - Take window of up to 128 tokens: `window = current_input[-128:]`
     - Construct input array: `arr = np.full((1, 128), tokenizer.pad_token_id, dtype=np.int32)`
     - Place tokens: `arr[0, :len(window)] = window`
     - Last valid token position: `last_valid = len(window) - 1`
     - Read logits: `pred[0, last_valid, :]`
   - Handle empty prompt by supplying `[tokenizer.bos_token_id]`.
   - Ensure prompt decoding preserves the original prompt without duplicating or omitting tokens.
   - Stop generation immediately on `EOS_ID=3`.
   - Guarantee `PAD_ID=0` and `BOS_ID=2` are never sampled or emitted into visible text.

### Phase 4: Comprehensive Test Suite
1. Create `tests/test_v3_integration.py` covering:
   - SHA-256 hash validation against `model_v3.json`.
   - Tokenizer loading, vocabulary size (8,000), special token IDs (0, 1, 2, 3).
   - Model input/output shape verification `(1, 128)` -> `(1, 128, 8000)`.
   - Valid token ID output ranges.
   - Last valid position calculation for single-token, short, normal, 128-token, and >128-token prompts.
   - EOS termination and PAD/BOS exclusion.
   - Temperature, Top-K, Top-P, and repetition penalty behavior.
   - Missing/corrupt v3 handling and rollback to v2.
   - API endpoint response validation (`/api/v1/generate`, `/api/v1/model/status`, `/api/v1/model/metrics`).
2. Run full pytest suite across `backend/tests/`, `ml/tests/`, and `tests/`.

### Phase 5: Frontend Verification
1. Test frontend API client and types with v3 responses.
2. Run `npm --prefix frontend run build` to confirm production bundle compiles cleanly.

### Phase 6: Fixed 30-Prompt Comparative Evaluation
1. Run identical generation on 30 diverse prompts across both v2 and v3.
2. Record latency, distinct n-grams, repetition, coherence, and save results to `artifacts/reports/v2_vs_v3_generation_comparison.json` and Markdown table.
3. Determine whether v3 should be the default based on empirical results.

### Phase 7: Final Documentation & Reporting
1. Update `README.md` with v3 usage, configuration, and rollback instructions.
2. Produce `docs/V3_INTEGRATION_REPORT.md` documenting verification, test output, evaluation analysis, and deployment checklist.
