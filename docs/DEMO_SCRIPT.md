# Academic Demonstration Script (3-5 Minutes)

This script provides an exact step-by-step walkthrough for presenting the **Story Continuation Generator Using LSTM** to professors, evaluators, or viva panels.

---

### Step 1: State the Problem Context (0:00 - 0:30)
- **Say:**  
  *"Good morning/afternoon. Today I am demonstrating an MDM Foundation of Generative AI project: a locally trained Story Continuation Generator based on a 2-layer Recurrent Neural Network with Long Short-Term Memory (LSTM) cells."*
- **Explain:**  
  *"Our academic goal is to demonstrate genuine autoregressive sequence modeling. The system has learned statistical and sequential word transitions from a classic literary story corpus. Crucially, all generation is performed entirely by our local LSTM—there are zero external API calls or hidden pre-trained Transformer weights."*

---

### Step 2: Present the Architecture (0:30 - 1:00)
- **Say:**  
  *"Before running the interface, here is our architecture: input sequences of 50 tokens pass through a 128-dimensional embedding layer, into two stacked LSTM layers of 256 units each with 20% dropout, and project through a dense layer into a softmax distribution across a 5,044-word vocabulary. The network contains approximately 3.2 million trainable parameters."*
- **Highlight Leakage Prevention:**  
  *"Data leakage was prevented by splitting whole stories at the document level—80% train, 10% validation, 10% test—before extracting sliding window pairs."*

---

### Step 3: Launch the Demonstration Interface (1:00 - 1:20)
- **Action:**  
  Open terminal and run:
  ```powershell
  python -m streamlit run app/app.py
  ```
- **Show:**  
  Point to the browser window. Show the sidebar with hardware specs (ASUS Zenbook 14 OLED, Intel Core Ultra 5 125H, TensorFlow CPU backend) and the 5 structured tabs.

---

### Step 4: Enter a Seed Prompt & Baseline Generation (1:20 - 2:00)
- **Action:**  
  Click on the preset prompt button **"🌲 The Dark Forest"** or type:
  `once upon a time in a dark and ancient forest`
- **Settings:**  
  Set **Temperature = 0.5**, **Top-k = 20**, **Seed = 42**.
- **Action:**  
  Click **"✨ Generate Continuation"**.
- **Explain:**  
  *"At a low temperature of 0.5 with top-k filtering, the model samples from high-probability transitions. Notice how grammatically coherent the story start is, utilizing typical fairy-tale vocabulary such as 'there lived', 'king', 'wood', and 'castle'."*

---

### Step 5: Modulate Temperature & Top-k to Demonstrate Sampling Dynamics (2:00 - 2:40)
- **Action:**  
  Increase **Temperature to 1.1** and set **Top-k = 50**.
- **Action:**  
  Click **"✨ Generate Continuation"**.
- **Explain:**  
  *"Notice how increasing temperature flattens the softmax distribution. The story continuation now introduces more diverse words and creative variance. Point out the lexical metrics displayed on screen: Distinct-1 and Distinct-2 scores increase, indicating higher vocabulary diversity."*

---

### Step 6: Review Training & Test Metrics (2:40 - 3:20)
- **Action:**  
  Click on the **"📊 Training & Metrics"** tab.
- **Explain:**  
  *"Here we observe the actual cross-entropy loss curve over training epochs. Both training and validation loss decreased steadily, showing healthy learning without divergence. We also evaluated our model on unseen test stories: our LSTM achieved a validation perplexity of ~50-60, significantly outperforming our smoothed Bigram baseline perplexity of ~200+."*

---

### Step 7: Explain an LSTM Core Concept & Close with Honest Limitations (3:20 - 4:00)
- **Action:**  
  Click on the **"📚 LSTM Methodology & Viva"** tab.
- **Explain:**  
  *"The reason an LSTM outperforms simpler Markov or vanilla RNN models is its gated Cell State ($C_t$). The Forget Gate $f_t$ allows it to selectively retain long-term narrative themes while the linear state update avoids vanishing gradients."*
- **State Limitations Honestly:**  
  *"To remain scientifically rigorous, we note key limitations: the context window is bounded at 50 tokens, rare words outside the 5,044 vocabulary map to unknown tokens, and because LSTMs compress memory into a single hidden vector without self-attention, long-range global plot consistency degrades over extended generations."*
- **Conclude:**  
  *"Thank you. I am ready for any questions from the panel."*
