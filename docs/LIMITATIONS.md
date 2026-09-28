# Academic Limitations & Boundary Analysis

To ensure strict academic integrity and scientifically defensible reporting, this document outlines the fundamental constraints, failure modes, and theoretical boundaries of the **Story Continuation Generator Using LSTM**.

---

## 1. Context Window & Sequential Compression Bottleneck

- **Context Horizon:** The model processes a maximum sequence window of $L = 50$ tokens. While adequate for capturing local phrases, clause coordination, and immediate character actions, narrative context established more than 50 words prior is not retained in the input window.
- **Hidden State Compression:** Even when processing sequences, an LSTM must compress all historical context into a fixed-dimensional continuous vector ($\mathbf{h}_t \in \mathbb{R}^{256}$). This sequential bottleneck inevitably induces informational decay over extended narrative arcs, unlike modern Transformer architectures that employ non-lossy pairwise self-attention over thousands of tokens.

---

## 2. Vocabulary Coverage & OOV (Out-of-Vocabulary) Handling

- **Vocabulary Truncation:** To ensure fast training and responsive local CPU inference, the vocabulary is bounded to the top 5,044 most frequent tokens in the fairy tale corpus.
- **Handling of Unseen Words:** Any word not present in this dictionary (such as modern terminology, specialized nouns, or rare character names) is mapped to the `<UNK>` token (index 1). Consequently, user prompts containing unfamiliar words will not benefit from specific lexical semantics for those tokens.

---

## 3. Repetition Dynamics Under Low-Entropy Sampling

- **Greedy Argmax Degeneracy:** When temperature is set near zero ($T \le 0.1$), the sampling engine collapses to deterministic argmax selection. Recurrent language models are prone to entering cyclic loops under greedy decoding (e.g., repeatedly generating "and the king said that the king said...").
- **Mitigation:** The system provides configurable **Top-$k$ filtering**, **Top-$p$ nucleus sampling**, and an explicit **Repetition Penalty** ($\alpha = 1.15$). Users should maintain temperature between 0.6 and 0.9 for balanced narrative flow.

---

## 4. Thematic vs. Syntactic Coherence

- **Local Syntactic Fluency:** The trained LSTM excels at producing syntactically plausible sentence fragments conforming to classic English folk tale style (verb-subject agreement, common prepositions, adjective-noun pairing).
- **Global Thematic Drift:** Because the model generates one token at a time without multi-level discourse planning, it lacks long-term story goal orientation. Characters introduced in an opening clause may disappear or shift roles over a multi-paragraph continuation.

---

## 5. Hardware & CPU Training Realities

- **Hardware Profile:** All training was performed locally on the 14-core Intel Core Ultra 5 125H processor of the ASUS Zenbook 14 OLED.
- **Resource Consciousness:** To prevent thermal throttling and ensure reproducible execution within 16 GB RAM limits, the batch size (64), sequence length (50), and corpus size (~130,000 tokens) were deliberately calibrated for sustainable CPU training rather than pushing for unnecessary parameters.
