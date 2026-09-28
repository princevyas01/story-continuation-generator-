# Viva Voce & Technical Defense Guide: Story Continuation LSTM

This document contains concise, technically rigorous answers to all potential oral examination and defense questions for the **Story Continuation Generator Using LSTM**.

---

### Q1. What is Generative AI?
**Answer:** Generative AI refers to probabilistic computational models capable of synthesizing novel content (such as text, images, or audio) by estimating and sampling from the learned probability distribution $P(X)$ or conditional distribution $P(X \mid Y)$ of the underlying training data, rather than merely mapping inputs to fixed class labels as in discriminative models.

---

### Q2. Why is this project considered generative?
**Answer:** The project models the conditional language distribution $P(w_t \mid w_{<t})$ over a discrete vocabulary of story words. Given an arbitrary seed prompt, the system iteratively samples and emits new words autoregressively, composing continuous, previously unseen story text rather than selecting from a canned template.

---

### Q3. What is a Recurrent Neural Network (RNN)?
**Answer:** An RNN is a neural network architecture designed for sequence modeling that maintains an internal hidden state vector $\mathbf{h}_t$. At each step $t$, $\mathbf{h}_t = \tanh(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b})$, allowing previous temporal context to influence current computations.

---

### Q4. Why is a vanilla RNN difficult to train on long-term dependencies?
**Answer:** When computing gradients via Backpropagation Through Time (BPTT), the gradient of the loss at step $T$ with respect to state $\mathbf{h}_t$ involves repeated multiplication by the transition weight matrix:
$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}}$$
If the eigenvalues of $\mathbf{W}_h$ are less than 1, the gradient exponentially approaches zero (**vanishing gradients**); if greater than 1, it diverges to infinity (**exploding gradients**), making long-range credit assignment impossible.

---

### Q5. What is an LSTM?
**Answer:** A Long Short-Term Memory (LSTM) network is a gated recurrent architecture introduced by Hochreiter & Schmidhuber (1997) that solves vanishing gradients by separating the recurrent state into a linear **Cell State** ($\mathbf{c}_t$) acting as an error carousel and a gated **Hidden State** ($\mathbf{h}_t$).

---

### Q6. What are the gates in an LSTM and their individual roles?
**Answer:**
1. **Forget Gate ($\mathbf{f}_t$):** Employs a sigmoid activation $\sigma \in (0, 1)$ to determine what fraction of the old cell state to discard.
2. **Input Gate ($\mathbf{i}_t$):** Modulates what proportion of the candidate memory $\tilde{\mathbf{c}}_t = \tanh(\dots)$ to write into the cell state.
3. **Output Gate ($\mathbf{o}_t$):** Filters the activated cell state $\tanh(\mathbf{c}_t)$ to emit the hidden state $\mathbf{h}_t$ used for predictions and subsequent time steps.

---

### Q7. What is an embedding?
**Answer:** An embedding is a trainable parameter lookup matrix $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d_e}$ that maps discrete, orthogonal one-hot token representations into a dense, low-dimensional, continuous vector space ($d_e = 128$) where semantic affinities between words can be captured via cosine similarity.

---

### Q8. What is tokenization?
**Answer:** Tokenization is the algorithmic process of parsing raw, unstructured textual strings into discrete semantic units (tokens) that map to integer identifiers within a fixed dictionary (vocabulary $\mathcal{V}$).

---

### Q9. Why use word-level tokens instead of character-level or subword BPE tokens?
**Answer:**
- Word-level modeling directly aligns with syntactic and semantic linguistic units, allowing meaningful word continuations with modest sequence lengths ($L = 50$).
- Character-level models require sequence lengths 5–7x longer to capture equivalent semantic context, heavily increasing recurrent latency.
- Subword (BPE) tokenizers add token splitting complexity that is unnecessary for a controlled, classical story corpus where a ~5,000 word vocabulary covers >95% of word tokens.

---

### Q10. How is the training target created?
**Answer:** Training targets are constructed via a sliding window of length $L = 50$:
$$\mathbf{X}_i = [w_i, w_{i+1}, \dots, w_{i+L-1}], \quad y_i = w_{i+L}$$
The input is the previous 50 word IDs, and the target is the single subsequent integer word ID.

---

### Q11. What loss function is used?
**Answer:** **Sparse Categorical Cross-Entropy**:
$$\mathcal{L} = - \log P(w_t = y_t \mid w_{<t})$$
The "sparse" variant takes integer class indices directly as targets rather than expanding them into dense one-hot vectors, saving substantial memory during training on 16GB RAM.

---

### Q12. What is Perplexity (PPL)?
**Answer:** Perplexity is the exponentiated average cross-entropy loss:
$$\text{PPL} = \exp(\mathcal{L})$$
It can be interpreted as the weighted average branching factor of the model—i.e., at each time step, the model is as uncertain as if it were choosing uniformly among $\text{PPL}$ candidate words. Lower perplexity indicates superior predictive confidence.

---

### Q13. What does temperature do during sampling?
**Answer:** Temperature $T > 0$ divides the logits prior to the softmax function:
$$P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- $T \to 0$: Probability mass concentrates entirely on the top token (deterministic greedy argmax), reducing grammar errors but increasing repetition.
- $T > 1.0$: Flattens the probability distribution across all words, encouraging lexical diversity and surprise at the expense of grammatical structure.

---

### Q14. What is top-k sampling?
**Answer:** Top-$k$ sampling restricts candidate selection to the $k$ most probable tokens at each generation step. The remaining vocabulary is assigned zero probability, preventing the generator from selecting erratic words from the low-probability tail.

---

### Q15. What is top-p (nucleus) sampling?
**Answer:** Nucleus sampling dynamically filters candidate tokens by choosing the smallest set of words whose cumulative probability mass exceeds threshold $p$ (e.g., $p = 0.90$). The candidate pool expands when uncertainty is high and contracts when certainty is high.

---

### Q16. Why not use a Transformer instead of an LSTM?
**Answer:**
1. **Academic Constraint:** The MDM syllabus specifically mandates understanding and building recurrent neural architectures to master sequential backpropagation.
2. **Computational Footprint:** Training a Transformer requires multi-head quadratic attention matrices ($O(L^2)$) and millions of parameters, which is ill-suited for fast local CPU training on a 16 GB laptop compared to a lightweight 2-layer LSTM ($O(L)$ computation).
3. **Pedagogical Contrast:** Master foundational recurrent memory mechanics first before progressing to self-attention.

---

### Q17. What are the limitations of this model?
**Answer:**
1. **Context Window Constraint:** The context window is fixed to 50 tokens; events described 100 words prior are lost.
2. **Vocabulary Coverage:** Rare words outside the top 5,044 vocabulary map to `<UNK>`.
3. **Thematic Coherence:** LSTMs lack global attention heads, so while local sentence grammar is preserved, complex overarching story plots may wander.

---

### Q18. How was data leakage avoided?
**Answer:** Data was partitioned strictly at the **story/document level** (80% train, 10% validation, 10% test) before sequence creation. If sliding windows were randomly split, adjacent overlapping windows from the same story would reside in both train and test splits, causing artificial test performance inflation.

---

### Q19. How was the model evaluated?
**Answer:**
1. **Quantitative Metrics:** Held-out test cross-entropy loss, test perplexity, Distinct-1/2 lexical diversity, and repeated 3-gram ratio.
2. **Empirical Baseline:** Benchmarked against a smoothed Bigram Markov model.
3. **Inference Latency:** Measured wall-clock time per token (ms/token).
4. **Qualitative Verification:** Standardized 20-sample human evaluation worksheet assessing coherence, grammaticality, and relevance.

---

### Q20. How does the Streamlit UI call the model?
**Answer:** Streamlit imports `src.generate.generate_continuation` directly. Model weights and the tokenizer are cached in memory using `@st.cache_resource`, ensuring inference occurs in memory without reloading disk files on user slider adjustments.

---

### Q21. What happens when the vocabulary is too large?
**Answer:** An excessively large vocabulary (e.g. 50,000 words) inflates the dense output projection layer $\mathbf{W}_p \in \mathbb{R}^{256 \times 50000}$ (12.8M parameters alone), significantly increasing memory usage, slowing down the final softmax normalization on CPU, and causing data sparsity where infrequent words are never sufficiently trained.

---

### Q22. What happens when a prompt contains unknown words?
**Answer:** Words outside the fitted vocabulary are mapped to the `<UNK>` token (index 1). The model has learned embedding representations for `<UNK>` during training, allowing it to process the sequence gracefully without crashing.

---

### Q23. Why does generated text sometimes repeat?
**Answer:** In autoregressive models, if the highest-probability token conditioned on context $[w_{t-1}, w_t]$ happens to trigger a cyclic sequence (e.g. "and the king said to the king"), the model falls into a degenerate feedback loop. Our generation engine mitigates this by applying a **repetition penalty** ($\alpha = 1.15$) and top-$p$/top-$k$ stochastic sampling.
