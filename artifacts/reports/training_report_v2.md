# Story Continuation Generator v2 — Training & Evaluation Report

**Date:** 2026-10-19 21:52:10  
**Target System:** Intel Core Ultra 5 125H, 16 GB RAM, Windows 11 (Pure CPU Execution)  

## 1. Executive Summary

The Story Continuation Generator was systematically re-architected from the baseline word-level, many-to-one LSTM into a subword Byte-Level BPE Many-to-Many autoregressive model with masked cross-entropy loss and logit-space nucleus sampling.

### Quantitative Head-to-Head Comparison

| Metric | Baseline (v1) | Upgraded Candidate (v2) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Architecture** | Stacked LSTM Many-to-One | Stacked LSTM Many-to-Many | Full sequence supervision |
| **Tokenizer** | Word-level (5,044 vocab) | Byte-Level BPE (8,000 vocab) | Zero OOV / Subword coverage |
| **Parameters** | 2,861,492 | 4,577,088 | Scaled capacity |
| **Test Loss** | 5.218 | 4.178 | Lower is better |
| **Test Perplexity** | 184.51 | 65.62 | **64.4% reduction** |
| **Next-Token Accuracy** | 19.28% | 27.47% | **+8.19% gain** |
| **Distinct-1 (Unigrams)** | 0.2510 | 0.2698 | +0.0188 diversity |
| **Distinct-2 (Bigrams)** | 0.7082 | 0.7050 | +-0.0031 diversity |
| **Repetition-4 Rate** | 0.0000 | 0.0000 | -0.0000 repetition |
| **CPU Latency per Token** | 7.5 ms | 21.2 ms | Interactive real-time |

## 2. Sample Generation Comparisons

### Prompt 1: Standard Fairytale Opening
> **Seed Prompt:** "*Once upon a time, deep inside an ancient forest, lived a tiny squirrel who loved*"

- **v1 Baseline:** her, the husband said,' i will have been seen to a work, and took me that it saw his porch fleecy, and when the man was no, and as the mother had not go to the hand, and the fox were so all
- **v2 Candidate:**  to the little boy.  The girl was scared and said, "I have your eyes!" she says. It replied!  "You can go, "What's you."  Lily was very happy to the dog of the mom

### Prompt 2: Standard Fairytale Opening
> **Seed Prompt:** "*Long ago in a kingdom surrounded by misty blue mountains, a young prince*"

- **v1 Baseline:** , and said:' cried the old woman, but he must eat him, and if it was not better, and she had seen a bag for it, and said,' t have not be seen to see them, for all that you is
- **v2 Candidate:** .  From the end, and had to explore it at him.

### Prompt 3: Standard Fairytale Opening
> **Seed Prompt:** "*In a peaceful village where flowers bloomed all year, there was a little house where*"

- **v1 Baseline:** he had been seen to do not see it.' then she will eat her all with her, and the bear,' said the king,' no, in that she shall be a daughter, and gave her one, and set on her head
- **v2 Candidate:**  she's dad would go back with the park. He felt happy and gave her more proud of a little big tree! She said, "I want to be better."  The little girl was so surprised and went to a moment and went away

### Prompt 4: Standard Fairytale Opening
> **Seed Prompt:** "*Once upon a time, an old clockmaker found an unusual golden key inside*"

- **v1 Baseline:** in the water' s father. they will go down to have the wall. the king was not all them, and as he had very beautiful things of our courtyard where he were.'' what that the princess,' cried the mother,'
- **v2 Candidate:** .

### Prompt 5: Character Introduction
> **Seed Prompt:** "*Lily was a cheerful five-year-old girl who always wore bright yellow rain boots because*"

- **v1 Baseline:** her father. and if he was going to a wine.' then he is not eat them, and the fox was sitting about all in his birds.' i should draw no one.' what are soon the servant and said,' said the
- **v2 Candidate:**  she smiled.  The little girl was so excited and had day. He had so much and said, "I will come away." She felt happy and said to listen that it's fun of them on. She had a happy bird.

