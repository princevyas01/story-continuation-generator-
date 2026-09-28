# Presentation Slide Deck Outline: Story Continuation Generator Using LSTM

A structured 12-slide presentation deck designed for academic evaluation, project review, and viva presentation.

---

### Slide 1: Title Slide
- **Title:** Story Continuation Generator Using an LSTM Language Model
- **Subtitle:** An Empirical Recurrent Generative AI Architecture for Autoregressive Text Generation
- **Course / Module:** MDM Foundation of Generative AI
- **Hardware Platform:** ASUS Zenbook 14 OLED (Intel Core Ultra 5 125H, Windows 11)
- **Author / Presenter:** Student Name / Roll Number

---

### Slide 2: Problem Statement & Motivation
- **Context:** Text continuation requires modeling non-linear, sequential dependencies in human language.
- **The Challenge:** Vanilla Markov chains lack memory; vanilla RNNs suffer from vanishing gradients; massive Transformer LLMs obscure core generative mechanisms and cannot be trained locally on consumer hardware.
- **Project Problem:** Design, train, and validate a fully transparent, locally runnable word-level generative neural language model from scratch on literary stories.

---

### Slide 3: Project Objectives
- Build a genuine word-level Recurrent Neural Network (LSTM) for next-token prediction.
- Implement strict document-level data partitioning to prevent data leakage.
- Develop a multi-control autoregressive sampling engine (Temperature, Top-$k$, Top-$p$, Repetition penalty).
- Conduct empirical evaluations comparing the LSTM against an N-gram baseline.
- Provide a clean Streamlit demonstration web interface with real-time inference statistics.

---

### Slide 4: Theoretical Foundations: From RNN to LSTM
- **Sequential Recurrence:** Passing hidden state $\mathbf{h}_t$ through time steps.
- **The Vanishing Gradient Bottleneck:** Continuous matrix multiplication causes $\lim_{T \to \infty} \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_0} = 0$.
- **LSTM Resolution:** Constant error carousel via the linear **Cell State** ($\mathbf{c}_t$).
- **Gating Equations:**
  - Forget Gate: $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$
  - Input Gate: $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$
  - Output Gate: $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$

---

### Slide 5: System Architecture & Component Topology
- Visual end-to-end pipeline diagram:
  - Raw Story Text $\to$ Normalization $\to$ Document Split $\to$ Tokenizer Fit $\to$ Sliding Window Sequences $(X, y)$.
  - Embedding $(128) \to$ Stacked LSTM $(256 \times 2) \to$ Softmax Head $(5044) \to$ Checkpoints.
  - Autoregressive Generation $\to$ Streamlit User Interface.

---

### Slide 6: Dataset Provenance & Leakage Prevention
- **Corpus:** Curated classic fairy tales from Project Gutenberg (Brothers Grimm & Hans Christian Andersen).
- **Licensing & Legal Compliance:** Public Domain in US and India (Rule of author's death + 60 years satisfied).
- **Corpus Statistics:** 62 documents, 129,486 total tokens, 5,044 unique vocabulary words.
- **Crucial Engineering Safeguard:** Document-level split (49 train, 6 val, 7 test) performed **before** sequence sliding window extraction to guarantee zero data leakage between train and test.

---

### Slide 7: Model Specifications & Hyperparameters
- **Input Dimension:** Shape $(B, 50)$ integer token IDs.
- **Embedding Layer:** $5,044 \times 128 = 645,632$ parameters.
- **Recurrent Depth:** 2 Stacked LSTM layers with 256 hidden units each.
- **Regularization:** 20% Dropout on recurrent projections.
- **Output Projection:** Dense Softmax layer over 5,044 vocabulary classes.
- **Total Parameters:** ~3,222,000 trainable parameters.
- **Optimization:** Adam ($\alpha = 0.002$), Gradient Clipping ($\text{clipnorm} = 1.0$), Sparse Categorical Cross-Entropy.

---

### Slide 8: The Autoregressive Generation Engine
- **Step-by-step Generation Algorithm:**
  1. Input prompt normalization & token encoding.
  2. Window truncation & left-padding to 50 tokens.
  3. Model forward pass generating next-token probability distribution.
  4. Temperature scaling ($T$): alters distribution entropy.
  5. Top-$k$ filtering: eliminates low-probability tail tokens.
  6. Top-$p$ nucleus filtering: cumulative probability thresholding.
  7. Repetition penalty: discounts probability of recently emitted tokens.
  8. Append selected token and repeat until `<END>` or max length.

---

### Slide 9: Experimental Results & Comparative Evaluation
- **Convergence Metrics:**
  - Training Loss progression & cross-entropy reduction across epochs.
  - Held-out test loss & test perplexity ($\text{PPL} = \exp(\mathcal{L})$).
- **Baseline Comparison:**
  - Smoothed Bigram Markov baseline vs. 2-layer LSTM language model.
  - Demonstration of recurrent memory advantage over single-token Markov transitions.
- **Lexical Quality Scores:** Distinct-1 (unigram diversity) and Distinct-2 (bigram diversity).
- **Hardware Efficiency:** CPU inference latency measured in milliseconds per token.

---

### Slide 10: Streamlit Demonstration Interface
- **Interactive UI Capabilities:**
  - Interactive prompt input with 4 instant fairy-tale preset buttons.
  - Dynamic sliders for Temperature, Top-$k$, Top-$p$, Repetition penalty, and Length.
  - Deterministic random seed switch for reproducible output verification.
  - Story output display with distinct styling for prompt vs. continuation.
  - Live performance telemetry badges (latency, tokens, distinct-N).
  - Single-click `.txt` file export.

---

### Slide 11: Scientific Limitations & Ethical Integrity
- **Model Boundaries:**
  - 50-token context horizon (catastrophic forgetting over long multi-paragraph stories).
  - Fixed 5,044-word vocabulary (infrequent proper nouns map to `<UNK>`).
  - Absence of global attention mechanisms (local grammar is strong, global thematic logic may drift).
- **Integrity Statement:**
  - No external commercial LLM APIs used.
  - No simulated or fabricated metrics—all numbers derived from local execution logs.

---

### Slide 12: Conclusion & References
- **Summary:** Successfully implemented, trained, evaluated, and deployed a local word-level LSTM generative language model on Windows hardware.
- **Key Takeaway:** Demonstrates the foundational principles of generative sequence modeling and stochastic sampling without massive cloud compute.
- **References:**
  1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
  2. TensorFlow & Keras Official Documentation (2026).
  3. Project Gutenberg Public Domain Literary Archives.
