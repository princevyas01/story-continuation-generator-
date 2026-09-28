# Model Training Report

- **Execution Mode:** Full Training Run
- **Timestamp:** 2026-09-28T19:24:15.751398
- **Dataset:** Project Gutenberg Classic Fairy Tales (Grimm, Andersen, Aesop)
- **Documents (Train/Val/Test):** 49 / 6 / 7
- **Corpus Total Tokens:** 129486
- **Vocabulary Size:** 5044 tokens
- **Sequence Length (Context Window):** 50
- **Model Architecture:** 2-Layer Word-Level LSTM (Units: 256, Embedding: 128, Dropout: 0.2)
- **Trainable Parameters:** 2,861,492
- **Batch Size:** 64
- **Epochs Requested / Completed:** 10 / 7
- **Final Training Loss:** 3.7991 (Perplexity: 44.66)
- **Best Validation Loss:** 5.0532 (Perplexity: 156.53)
- **Held-Out Test Loss:** 5.2177 (Perplexity: 184.51)
- **Training Duration:** 1375.16 seconds (22.92 minutes)
- **Platform / Machine:** Intel Core Ultra 5 125H (14 cores / 18 threads), Windows 11, TensorFlow CPU (oneDNN enabled)
