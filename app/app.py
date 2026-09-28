"""Streamlit Demonstration App for Story Continuation Generator Using LSTM."""

import os
import sys
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ui_helpers import load_cached_resources, load_metrics_and_artifacts, load_precomputed_samples
from src.generate import generate_continuation
from src.metrics import evaluate_text_generation_quality

st.set_page_config(
    page_title="Story Continuation Generator | LSTM",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic Polish
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .story-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-family: Georgia, serif;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .prompt-highlight {
        color: #1E3A8A;
        font-weight: 600;
    }
    .continuation-highlight {
        color: #0F172A;
    }
    .metric-badge {
        display: inline-block;
        background: #E2E8F0;
        color: #334155;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration & Hardware Status
st.sidebar.title("🎛️ Generation Controls")

# Load Resources
model, tokenizer, cfg, load_error = load_cached_resources()
artifacts = load_metrics_and_artifacts()
metrics = artifacts.get("metrics", {})
manifest = artifacts.get("manifest", {})

if load_error:
    st.error(f"⚠️ {load_error}")
    st.info("The model is currently training or needs to be trained. Please wait for training to finish.")

# Sidebar Sliders
st.sidebar.markdown("### Sampling Hyperparameters")
temperature = st.sidebar.slider(
    "Temperature (Softmax Sharpness)",
    min_value=0.1,
    max_value=1.5,
    value=float(cfg.get("generation", {}).get("default_temperature", 0.8)),
    step=0.05,
    help="Controls stochasticity. Lower values (0.2 - 0.5) produce safe, frequent words. Higher values (0.8 - 1.2) introduce diversity."
)

top_k = st.sidebar.slider(
    "Top-k Candidates",
    min_value=0,
    max_value=100,
    value=int(cfg.get("generation", {}).get("default_top_k", 40)),
    step=5,
    help="Restricts sampling to the k highest-probability tokens. Set to 0 to disable."
)

top_p = st.sidebar.slider(
    "Top-p (Nucleus Cutoff)",
    min_value=0.1,
    max_value=1.0,
    value=float(cfg.get("generation", {}).get("default_top_p", 0.90)),
    step=0.05,
    help="Limits sampling to the cumulative probability mass p. 1.0 keeps all candidates."
)

repetition_penalty = st.sidebar.slider(
    "Repetition Penalty",
    min_value=1.0,
    max_value=2.0,
    value=float(cfg.get("generation", {}).get("repetition_penalty", 1.15)),
    step=0.05,
    help="Penalizes tokens that have appeared recently in the story."
)

max_tokens = st.sidebar.slider(
    "Continuation Length (Tokens)",
    min_value=10,
    max_value=120,
    value=int(cfg.get("generation", {}).get("default_max_new_tokens", 50)),
    step=5
)

use_seed = st.sidebar.checkbox("Deterministic Seed (Reproducibility)", value=True)
seed_value = None
if use_seed:
    seed_value = st.sidebar.number_input("Seed Value", min_value=0, max_value=999999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💻 Local Compute Node")
st.sidebar.markdown("**Processor:** Intel Core Ultra 5 125H (14C / 18T)")
st.sidebar.markdown("**Hardware Tier:** ASUS Zenbook 14 OLED")
st.sidebar.markdown("**Execution Backend:** TensorFlow 2.21 CPU (oneDNN)")
st.sidebar.markdown("**External LLMs:** None (Purely Local LSTM)")

# Tabs Layout
tab_gen, tab_info, tab_metrics, tab_samples, tab_methodology = st.tabs([
    "✍️ Story Generator",
    "ℹ️ Model Specs",
    "📊 Training & Metrics",
    "🧪 Sampling Sweeps",
    "📚 LSTM Methodology & Viva"
])

# ==========================================
# TAB 1: STORY GENERATOR
# ==========================================
with tab_gen:
    st.markdown("<div class='main-title'>Story Continuation Generator</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>A locally trained 2-layer Word-Level LSTM Neural Language Model. "
        "Predicts next tokens autoregressively without external APIs or Transformers.</div>",
        unsafe_allow_html=True
    )

    # Example prompts helper
    st.markdown("**Quick Preset Prompts:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    preset_prompt = None
    if col_p1.button("🌲 The Dark Forest"):
        preset_prompt = "once upon a time in a dark and ancient forest"
    if col_p2.button("👑 The Golden Goose"):
        preset_prompt = "there was once a simple man who found a golden goose in the woods"
    if col_p3.button("🐸 The Frog Prince"):
        preset_prompt = "the youngest princess sat by the fountain and threw her golden ball"
    if col_p4.button("🧙 The Hidden Cave"):
        preset_prompt = "deep inside the secret mountain the old king heard a strange sound"

    # Story input area
    prompt_text = st.text_area(
        "Enter your story prefix:",
        value=preset_prompt if preset_prompt else "once upon a time in a peaceful little village",
        height=100,
        help="Type the beginning of a story. The LSTM will predict the subsequent tokens."
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    gen_clicked = col_btn1.button("✨ Generate Continuation", type="primary", use_container_width=True)
    clear_clicked = col_btn2.button("🗑️ Clear Output", use_container_width=True)

    if clear_clicked:
        st.session_state["last_generation"] = None

    if gen_clicked:
        if not model or not tokenizer:
            st.error("Model or tokenizer is not ready. Please ensure training is completed.")
        elif not prompt_text.strip():
            st.warning("Please enter a valid story prompt.")
        else:
            with st.spinner("Generating autoregressive continuation token-by-token..."):
                seq_len = cfg.get("tokenization", {}).get("sequence_length", 50)
                gen_result = generate_continuation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_text,
                    seq_len=seq_len,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    seed=seed_value if use_seed else None
                )
                quality = evaluate_text_generation_quality(gen_result["continuation"])
                gen_result["quality"] = quality
                st.session_state["last_generation"] = gen_result

    # Display Generation Output
    if st.session_state.get("last_generation"):
        res = st.session_state["last_generation"]
        st.markdown("### 📖 Generated Story Output")
        
        story_html = f"""
        <div class='story-card'>
            <span class='prompt-highlight'>{res['prompt']}</span>
            <span class='continuation-highlight'> {res['continuation']}</span>
        </div>
        """
        st.markdown(story_html, unsafe_allow_html=True)

        # Performance & Diversity Badges
        q = res["quality"]
        st.markdown(f"""
        <div style='margin-top: 12px;'>
            <span class='metric-badge'>⏱️ Latency: {res['latency_ms_per_token']:.1f} ms/token</span>
            <span class='metric-badge'>🔢 Tokens Generated: {res['tokens_generated']}</span>
            <span class='metric-badge'>🔤 Distinct-1: {q['distinct_1']:.2f}</span>
            <span class='metric-badge'>👥 Distinct-2: {q['distinct_2']:.2f}</span>
            <span class='metric-badge'>🔄 3-gram Repetition: {q['repetition_ratio_3gram']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="💾 Download Generated Story (.txt)",
            data=f"Prompt:\n{res['prompt']}\n\nContinuation:\n{res['continuation']}\n\nFull Story:\n{res['full_story']}",
            file_name="generated_story.txt",
            mime="text/plain"
        )

# ==========================================
# TAB 2: MODEL SPECS & ARCHITECTURE
# ==========================================
with tab_info:
    st.markdown("## ℹ️ LSTM Model Architecture & Specifications")
    st.markdown("Detailed breakdown of the recurrent neural language model trained on this machine.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Model Architecture", "2-Layer LSTM")
    col_m2.metric("Vocabulary Size", f"{tokenizer.vocab_size if tokenizer else 5044:,} words")
    col_m3.metric("Context Window", f"{cfg.get('tokenization', {}).get('sequence_length', 50)} tokens")
    col_m4.metric("Trainable Parameters", f"{model.count_params():,}" if model else "~3.2M")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Embedding Dim", f"{cfg.get('model', {}).get('embedding_dim', 128)}")
    col_s2.metric("LSTM Units / Layer", f"{cfg.get('model', {}).get('lstm_units', 256)}")
    col_s3.metric("Dropout Rate", f"{cfg.get('model', {}).get('dropout_rate', 0.2)}")
    col_s4.metric("Gradient Clipnorm", f"{cfg.get('model', {}).get('clipnorm', 1.0)}")

    st.markdown("### 📐 Layer Composition Diagram")
    st.code("""
    Layer 0: Input Sequence              (batch_size, 50)           int32
    Layer 1: Embedding                   (batch_size, 50, 128)      float32
    Layer 2: LSTM (return_seq=True)      (batch_size, 50, 256)      float32 (Dropout: 0.2)
    Layer 3: LSTM (return_seq=False)     (batch_size, 256)          float32 (Dropout: 0.2)
    Layer 4: Dense (Softmax)             (batch_size, Vocab_Size)   float32
    """, language="text")

    st.markdown("### 📜 Dataset Provenance & Licensing")
    st.markdown(f"- **Source:** {manifest.get('source_name', 'Project Gutenberg Classic Fairy Tales')}")
    st.markdown(f"- **License:** {manifest.get('license_name', 'Public Domain (Life + 60 Years Compliant)')}")
    st.markdown(f"- **Total Documents:** {manifest.get('document_count', 62)} tales")
    st.markdown(f"- **Total Tokens:** {manifest.get('token_count', 129486):,} words")
    st.markdown(f"- **Leakage Prevention Split:** Document-level partition ({manifest.get('train_document_count', 49)} train / {manifest.get('validation_document_count', 6)} val / {manifest.get('test_document_count', 7)} test)")

# ==========================================
# TAB 3: TRAINING & METRICS
# ==========================================
with tab_metrics:
    st.markdown("## 📊 Training History & Validation Performance")

    bdir = cfg.get("base_dir", "E:\\mdm")
    plot_path = os.path.join(bdir, "artifacts", "plots", "loss_curve.png")
    
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Cross-Entropy Loss Progression Over Training Epochs", use_container_width=True)
        else:
            st.info("Loss curve plot artifact will appear here upon completion of training.")

    with col_t2:
        st.markdown("### 🎯 Benchmark Metrics")
        st.markdown(f"- **Training Loss:** `{metrics.get('train_loss', 'N/A')}`")
        st.markdown(f"- **Validation Loss:** `{metrics.get('val_loss', 'N/A')}`")
        st.markdown(f"- **Held-Out Test Loss:** `{metrics.get('test_loss', 'N/A')}`")
        st.markdown(f"- **Training Perplexity:** `{metrics.get('train_perplexity', 'N/A')}`")
        st.markdown(f"- **Validation Perplexity:** `{metrics.get('val_perplexity', 'N/A')}`")
        st.markdown(f"- **Test Perplexity:** `{metrics.get('test_perplexity', 'N/A')}`")
        st.markdown(f"- **Baseline (Bigram) Perplexity:** `{metrics.get('bigram_baseline_perplexity', 'N/A')}`")
        st.markdown(f"- **Elapsed Training Time:** `{metrics.get('training_duration_seconds', 0):.1f} s` ({metrics.get('training_duration_seconds', 0)/60:.1f} min)")

    if artifacts.get("history_df") is not None:
        st.markdown("### 📋 Epoch-by-Epoch Progress Table")
        st.dataframe(artifacts["history_df"], use_container_width=True)

# ==========================================
# TAB 4: SAMPLING SWEEPS
# ==========================================
with tab_samples:
    st.markdown("## 🧪 Empirical Sampling Sweeps")
    st.markdown(
        "Explore how temperature ($T$) and Top-$k$ filtering systematically alter the predictability, "
        "lexical diversity, and repetition characteristics of the LSTM generator."
    )

    precomputed = load_precomputed_samples()
    if precomputed:
        df_samples = pd.DataFrame([
            {
                "Prompt": s["prompt"],
                "Temperature": s["temperature"],
                "Top-k": s["top_k"],
                "Distinct-1": s["metrics"]["distinct_1"],
                "Distinct-2": s["metrics"]["distinct_2"],
                "3-Gram Repetition": s["metrics"]["repetition_ratio_3gram"],
                "Latency (ms)": s["latency_ms_per_token"],
                "Continuation": s["continuation"]
            }
            for s in precomputed
        ])
        st.dataframe(df_samples, use_container_width=True)

        st.markdown("### 🔍 Side-by-Side Comparison Cards")
        for s in precomputed[:4]:
            with st.expander(f"Prompt: \"{s['prompt']}\" (T={s['temperature']}, k={s['top_k']})"):
                st.write(f"**Full Generated Text:** {s['full_story']}")
                st.write(f"**Metrics:** Distinct-1: `{s['metrics']['distinct_1']}` | Distinct-2: `{s['metrics']['distinct_2']}` | Latency: `{s['latency_ms_per_token']} ms/token`")
    else:
        st.info("Precomputed generation sweep samples will be displayed here once `src/evaluate.py` has run.")

# ==========================================
# TAB 5: METHODOLOGY & VIVA PREPARATION
# ==========================================
with tab_methodology:
    st.markdown("## 📚 Recurrent Neural Network Methodology & Viva Cheat Sheet")

    with st.expander("1. What is an LSTM and how does it solve the Vanishing Gradient problem?", expanded=True):
        st.markdown("""
        A **Long Short-Term Memory (LSTM)** network is a specialized recurrent neural network (RNN) architecture designed to process sequential data while mitigating the vanishing and exploding gradient problems inherent in standard vanilla RNNs.
        
        Standard RNNs compute hidden state via:
        $$h_t = \\tanh(W_h h_{t-1} + W_x x_t + b)$$
        Repeated backpropagation through time (BPTT) leads to continuous multiplication of weight matrices, causing gradients either to vanish to zero or explode exponentially over long sequences.
        
        The LSTM introduces a dedicated linear **Cell State ($C_t$)** acting as an information highway, regulated by three multiplicative gates:
        1. **Forget Gate ($f_t$):** Determines what proportion of past information to discard.
           $$f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$$
        2. **Input Gate ($i_t$):** Controls what new information is added to the cell state.
           $$i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$$
           $$\\tilde{C}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)$$
        3. **Cell State Update ($C_t$):** Linear combination preventing gradient decay.
           $$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$
        4. **Output Gate ($o_t$):** Determines the next hidden state.
           $$o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$$
           $$h_t = o_t \\odot \\tanh(C_t)$$
        """)

    with st.expander("2. Why is this project Generative AI?"):
        st.markdown("""
        **Generative AI** encompasses models that learn the underlying joint probability distribution $P(X)$ or conditional sequential distribution $P(x_t | x_{<t})$ of training data to synthesize novel, coherent artifacts.
        
        Unlike discriminative models that assign input text to fixed class labels ($P(Y|X)$), this LSTM operates autoregressively: given a prefix sequence of tokens, it outputs a probability distribution over the vocabulary and samples new tokens one step at a time, generating novel narrative continuations that did not exist in the training corpus.
        """)

    with st.expander("3. How do Temperature, Top-k, and Top-p Sampling influence generation?"):
        st.markdown("""
        - **Temperature ($T$):** Modulates the logit distribution before softmax:
          $$P(w_i) = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$
          When $T \\to 0$, distribution approaches a Dirac delta (greedy argmax), causing safe but repetitive phrases. When $T > 1$, probability mass flattens across rare words, increasing lexical diversity at the cost of grammatical coherence.
        - **Top-k Sampling:** Restricts the candidate pool to the top $k$ tokens with the highest probabilities, preventing the model from selecting bizarre or out-of-context tokens from the long tail.
        - **Top-p (Nucleus) Sampling:** Dynamically chooses the smallest set of words whose cumulative probability mass exceeds threshold $p$, allowing the candidate pool size to expand or contract based on model certainty.
        """)

    with st.expander("4. How was Data Leakage avoided?"):
        st.markdown("""
        In sequential text generation, splitting sliding window sequences $(X_i, y_i)$ randomly across train and test sets leads to **catastrophic data leakage**, because consecutive overlapping windows ($[w_1...w_{50}]$ and $[w_2...w_{51}]$) would be divided between train and validation.
        
        This project strictly implements **document-level partitioning**: entire stories are assigned to Train (80%), Validation (10%), or Test (10%) sets *before* sliding windows are generated. The model is evaluated strictly on unseen narratives.
        """)

    with st.expander("5. What are the fundamental limitations of an LSTM compared to Modern Transformers?"):
        st.markdown("""
        1. **Sequential Bottleneck:** LSTMs compute hidden states sequentially ($h_{t-1} \\to h_t$), preventing massive parallelization during training. Transformers process entire sequences simultaneously using self-attention.
        2. **Fixed Context Vector Compression:** While superior to vanilla RNNs, LSTMs must compress all past context into a fixed-size vector ($C_t, h_t$), leading to catastrophic forgetting over hundreds of tokens.
        3. **Context Sensitivity:** Self-attention computes dynamic pairwise affinities between all tokens in a context window, capturing complex syntactic relationships across arbitrary distances.
        """)
