# Technical References & Compatibility Audit

This document records the official documentation, specifications, framework APIs, and legal references utilized during the design and development of the **Story Continuation Generator Using LSTM**.

---

## 1. Primary Frameworks & Tooling

### TensorFlow & Keras
- **TensorFlow Pip Installation Guide:** [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip)  
  *Findings & Windows Architecture:* Starting after TensorFlow 2.10, native Windows GPU support was deprecated by TensorFlow in favor of WSL2. For standard Windows desktops and laptops without complex WSL2 virtualization, official TensorFlow CPU wheels (`tensorflow >= 2.16`, including `tensorflow 2.21`) run natively and stably on Windows AMD64 CPUs using standard oneDNN multi-threaded CPU instructions.
- **Keras 3 Multi-Backend API:** [https://keras.io/](https://keras.io/) / [https://keras.io/api/layers/recurrent_layers/lstm/](https://keras.io/api/layers/recurrent_layers/lstm/)  
  *Findings:* Keras 3 provides unified neural abstractions. The `keras.layers.LSTM` layer offers standard gated cell state updates with configurable recurrent dropout, unrolling, and sequences return.
- **Official Character/Word Language Modeling Guide:**  
  [https://keras.io/examples/generative/lstm_character_level_text_generation/](https://keras.io/examples/generative/lstm_character_level_text_generation/)  
  Demonstrates recurrent state progression and categorical temperature sampling.

### Streamlit Framework
- **Streamlit Installation & App Execution:** [https://docs.streamlit.io/get-started/installation](https://docs.streamlit.io/get-started/installation)  
  Official execution command: `python -m streamlit run app/app.py`  
  *Caching Patterns:* `st.cache_resource` is used for deep learning model and tokenizer weights to prevent expensive reloading across user interactions; `st.cache_data` is used for metrics and history tables.

---

## 2. Hardware Architecture: ASUS Zenbook 14 OLED

- **Processor:** Intel Core Ultra 5 125H (Meteor Lake architecture).
  - Physical Cores: 14 cores (4 Performance-cores, 8 Efficient-cores, 2 Low-Power Efficient cores).
  - Logical Threads: 18 threads.
  - Base Frequency: 1.2 GHz (P-core) / 700 MHz (E-core), Boost up to 4.5 GHz.
- **Graphics & AI Acceleration:**
  - Integrated GPU: Intel Arc Graphics (8 Xe-cores).
  - Integrated NPU: Intel AI Boost (2 Gen3 NPU engines).
- **RAM & Thermal Profile:**
  - 16 GB LPDDR5X onboard memory.
  - Thin-and-light chassis requires thermal-conscious training: bounded batch sizes (`batch_size=32` to `64`), sparse cross-entropy loss to prevent dense matrix explosion, and gradient clipping (`clipnorm=1.0`) to avoid numerical instability.

---

## 3. Dataset Provenance & Copyright Compliance

- **Project Gutenberg Policy & Terms:** [https://www.gutenberg.org/policy/license](https://www.gutenberg.org/policy/license)
  - Project Gutenberg distributes works that are in the public domain in the United States.
- **Jurisdiction Notice (India / International):**
  - Under the Indian Copyright Act, 1957 (Section 22), copyright protection for literary works expires 60 years after the end of the calendar year in which the author died ("Life + 60 years").
  - The corpus chosen for this academic generator comprises classic fairy tales and folk tales by authors whose works entered the public domain globally well before the required statutory period (e.g., Brothers Grimm, d. 1859/1863; Hans Christian Andersen, d. 1875; Aesop, antiquity).
  - All text files are downloaded, verified for licensing headers, and processed locally with auditable manifests.
