# Mathematical & Theoretical Methodology: Story Continuation LSTM

This document details the mathematical formalisms, gating mechanics, objective functions, and sampling algorithms underlying the **Story Continuation Generator Using LSTM**.

---

## 1. Problem Formulation: Autoregressive Language Modeling

Given a sequential narrative corpus represented as an ordered sequence of discrete word tokens:
$$\mathcal{W} = (w_1, w_2, \dots, w_T)$$

The objective of an autoregressive language model is to maximize the joint probability of the sequence, factorized using the chain rule of probability:
$$P(\mathcal{W}) = \prod_{t=1}^T P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

In this project, the conditioning history is truncated to a context window of length $L = 50$:
$$P(w_t \mid w_{<t}) \approx P(w_t \mid w_{t-L}, \dots, w_{t-1})$$

---

## 2. Text Representation & Tokenization

1. **Word-Level Vocabulary:**  
   The vocabulary $\mathcal{V}$ is constructed by selecting the top $|\mathcal{V}| = 5,044$ most frequent tokens from the training corpus, plus 4 reserved special tokens:
   $$\mathcal{V}_{\text{special}} = \{\langle\text{PAD}\rangle: 0, \, \langle\text{UNK}\rangle: 1, \, \langle\text{START}\rangle: 2, \, \langle\text{END}\rangle: 3\}$$
2. **Dense Vector Embedding:**  
   Each discrete token ID $w_t \in \{0, \dots, |\mathcal{V}|-1\}$ is mapped to a continuous dense vector representation via an embedding matrix $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d_e}$ where $d_e = 128$:
   $$\mathbf{x}_t = \mathbf{E}[w_t] \in \mathbb{R}^{128}$$

---

## 3. Long Short-Term Memory (LSTM) Internal Mechanics

Standard vanilla Recurrent Neural Networks suffer from vanishing and exploding gradients when backpropagating through time (BPTT). The LSTM resolves this by maintaining a constant error carousel via an internal **Cell State** ($\mathbf{c}_t$) modulated by three non-linear gates.

At time step $t$, with input vector $\mathbf{x}_t$ and previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^{d_h}$ ($d_h = 256$):

### A. The Forget Gate ($\mathbf{f}_t$)
Controls what fraction of the previous cell state memory $\mathbf{c}_{t-1}$ to discard:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$$
where $\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$ is the logistic sigmoid function.

### B. The Input Gate ($\mathbf{i}_t$) & Candidate State ($\tilde{\mathbf{c}}_t$)
Decides which new information from the current token should be stored in memory:
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$$
where $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$ generates candidate values.

### C. The Updated Cell State ($\mathbf{c}_t$)
Updates the internal cell state via an additive, linear combination:
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
where $\odot$ denotes the element-wise Hadamard product. Because addition is linear, error gradients can flow backwards through time without exponentially decaying to zero.

### D. The Output Gate ($\mathbf{o}_t$) & Hidden State ($\mathbf{h}_t$)
Filters the updated cell state to emit the current hidden representation:
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

In our 2-layer stacked architecture, layer 1 emits $(\mathbf{h}_1^{(1)}, \dots, \mathbf{h}_L^{(1)})$, which serves as input to layer 2, emitting the final context vector $\mathbf{h}_L^{(2)} \in \mathbb{R}^{256}$.

---

## 4. Next-Token Projection & Loss Optimization

### A. Softmax Distribution
The final hidden state is projected to the vocabulary dimension through a dense linear layer followed by the softmax activation:
$$\mathbf{z} = \mathbf{W}_p \mathbf{h}_L^{(2)} + \mathbf{b}_p \in \mathbb{R}^{|\mathcal{V}|}$$
$$P(w_{t} = j \mid w_{<t}) = \frac{\exp(z_j)}{\sum_{k=1}^{|\mathcal{V}|} \exp(z_k)}$$

### B. Sparse Categorical Cross-Entropy Loss
For true next-token label $y_t \in \{0, \dots, |\mathcal{V}|-1\}$, the objective is to minimize:
$$\mathcal{L}(\theta) = - \log P(w_t = y_t \mid w_{<t})$$

### C. Gradient Clipping
To prevent gradient explosions during training, gradient vectors $\mathbf{g}$ are clipped by norm:
$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \, \frac{\text{clipnorm}}{\|\mathbf{g}\|_2}\right), \quad \text{with } \text{clipnorm} = 1.0$$

---

## 5. Evaluation Metrics

### A. Perplexity (PPL)
Perplexity measures the effective branching factor / uncertainty of the model when predicting the next word:
$$\text{PPL} = \exp\left( \frac{1}{N} \sum_{i=1}^N \mathcal{L}_i \right) = \exp(\bar{\mathcal{L}})$$
A lower perplexity indicates the model assigns higher probability to the true test sequences.

### B. Distinct-$n$ Lexical Diversity
Measures the variety of words and phrases generated:
$$\text{Distinct-}n = \frac{|\text{Unique } n\text{-grams in output}|}{\text{Total } n\text{-grams in output}} \in [0.0, 1.0]$$

### C. Repetition Ratio
Measures repetitive loops in generated sequences:
$$\text{Repetition-}n = 1.0 - \text{Distinct-}n$$

---

## 6. Autoregressive Sampling Controls

1. **Temperature ($T$):**
   $$P_T(w_j) = \frac{\exp(z_j / T)}{\sum_k \exp(z_k / T)}$$
   - $T < 1.0$: Sharper distribution, conservative and frequent vocabulary.
   - $T > 1.0$: Flatter distribution, higher novelty and lexical entropy.
2. **Top-$k$ Truncation:**
   Keeps only the $k$ tokens with the highest probabilities, setting all others to zero before renormalizing:
   $$\mathcal{V}_{\text{top-k}} = \operatorname{argtop}_k(P_T)$$
3. **Top-$p$ (Nucleus) Truncation:**
   Dynamically determines the smallest subset $V^{(p)} \subset \mathcal{V}$ such that:
   $$\sum_{w \in V^{(p)}} P(w) \ge p$$
4. **Repetition Penalty ($\alpha$):**
   For tokens $w$ generated in the recent window, their logits/probabilities are discounted:
   $$P'(w) = P(w) / \alpha \quad (\alpha > 1.0)$$
