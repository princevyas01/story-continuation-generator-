# Project Report & Viva Examination Notes
## Story Continuation Generator using Stacked LSTM Language Model

---

### 1. Project Motivation & Problem Statement
Natural language generation (NLG) involves predicting sequential tokens conditioned on prior context. While modern multi-billion parameter transformer models dominate cloud APIs, they incur high memory footprints, latency overheads, and dependency on external servers. 
This project implements a **purely local, self-contained, CPU-efficient autoregressive language model** using a **2-layer stacked Long Short-Term Memory (LSTM)** network trained on classic English folklore and fairy tales (Grimm & Andersen). The user interface is built in **React + TypeScript + Vite** and communicates with a **FastAPI** backend serving single-server inference on an **Intel Core Ultra 5 125H** laptop.

---

### 2. Theoretical Architecture & Mathematical Formulation

#### 2.1 Why LSTM for Sequential Story Continuation?
Vanilla Recurrent Neural Networks (RNNs) suffer from vanishing and exploding gradients when backpropagating through time (BPTT), making them incapable of maintaining context beyond 5–10 words. LSTMs introduce an internal constant-error carousel (the cell state \(c_t\)) regulated by three multiplicative gating mechanisms:

1. **Forget Gate (\(f_t\)):**
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
   Controls what portion of past context to discard.
2. **Input Gate (\(i_t\)) & Candidate State (\(\tilde{c}_t\)):**
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
   Determines what new narrative information to store in the cell state.
3. **Cell State Update (\(c_t\)):**
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
4. **Output Gate (\(o_t\)) & Hidden State (\(h_t\)):**
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(c_t)$$

#### 2.2 Network Topology
- **Input Layer:** Sequence tensor of shape `(batch_size, 50)` containing integer token indices.
- **Embedding Layer:** \(5004 \times 128\) dense projection matrix mapping sparse token IDs to continuous representations.
- **LSTM Layer 1:** 256 hidden units, `return_sequences=True`, with 20% recurrent dropout.
- **LSTM Layer 2:** 256 hidden units, `return_sequences=False`, with 20% recurrent dropout.
- **Dense Head:** 5004 linear projection units with Softmax activation emitting categorical distribution \(P(w_{t+1} \mid w_1, \dots, w_t)\).
- **Total Parameters:** ~2,453,804 trainable weights.

---

### 3. Autoregressive Sampling Dynamics

During generation, next tokens are predicted autoregressively. To prevent monotonous repetition or degenerate loops, three sampling techniques are combined:

1. **Temperature Scaling (\(T\)):**
   $$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
   - \(T \to 0\): Greedy decoding (deterministic, can loop).
   - \(T \in [0.7, 0.9]\): Balanced narrative coherence and lexical novelty.
   - \(T > 1.2\): High entropy (creative but prone to syntactic degradation).

2. **Top-K Truncation:**
   Filters the categorical distribution to only the top \(K\) most probable vocabulary words before sampling, avoiding out-of-context words from the long tail.

3. **Nucleus / Top-P Sampling:**
   Selects the smallest subset of candidate words whose cumulative probability mass exceeds \(P\) (\(\sum_{i \in V^{(p)}} p_i \ge P\)), adapting the pool width dynamically based on prediction certainty.

4. **Repetition Penalty:**
   Penalizes tokens that appeared in the recent 30-token history window by dividing their probability logits by 1.15.

---

### 4. Quantitative Results & Evaluation Summary

| Metric | Measured Value | Meaning |
| :--- | :--- | :--- |
| **Best Validation Loss** | **5.0532** | Cross-entropy over 5,004 discrete vocabulary items |
| **Perplexity (PPL)** | **156.52** | Effective branching factor / word uncertainty |
| **Token Accuracy** | **23.14%** | Exact next-word prediction accuracy |
| **Inference Latency** | **~18 - 25 ms / token** | CPU execution on Intel Core Ultra 5 125H |
| **Vocabulary Size** | **5,004** | Top frequent words + 4 special tokens (`<PAD>`, `<UNK>`, `<START>`, `<END>`) |
| **Training Dataset** | **129,480 tokens** | 62 public-domain Grimm & Andersen fairy tales |

---

### 5. Viva / Defense Questions & Model Answers

**Q1: Why use an LSTM over a standard Markov chain or N-gram model?**  
*Answer:* An N-gram model has zero generalization across unseen phrases (data sparsity) and suffers from exponential parameter explosion with context length \(N\). An LSTM maps words into a continuous embedding space where semantically similar words share vector space, and retains variable-length dependencies through its cell state without requiring explicit \(N\)-gram counting tables.

**Q2: Why not just call an OpenAI or Claude API?**  
*Answer:* Using a commercial API reduces machine learning to a network request, completely bypassing core architectural understanding of recurrent units, backpropagation through time, embedding geometries, and sampling mathematics. Furthermore, a local model operates offline, costs zero API fees, guarantees complete data privacy, and requires predictable laptop resources.

**Q3: How does padding affect the loss during training?**  
*Answer:* During sequence batching, sequences are left-padded with the `<PAD>` token (index 0). During loss calculation, the model either masks out padding tokens or uses explicit sample weights so that loss gradients are only updated based on actual narrative word predictions.

**Q4: What causes the LSTM to repeat words or get stuck in loops?**  
*Answer:* At low temperatures or with pure greedy argmax decoding, the model selects the single highest-probability token. In natural language, common words like "the", "and", "king" frequently predict each other in loops. Introducing temperature scaling (\(T=0.8\)), top-k truncation, and repetition penalties breaks cyclic attractors.
