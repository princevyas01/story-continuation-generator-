# Story LSTM v3 Training Report

## Data
- Source: `roneneldan/TinyStories`
- Selected stories: 50000
- Document split: {'seed': 42, 'source_sha256': '618d76b2f716b2d71b142efffdb52815af581c8020602401858436a1d4ab7dbe', 'split_policy': '90/5/5 document-level split before windowing', 'train_stories': 45000, 'validation_stories': 2500, 'test_stories': 2500, 'exact_document_overlap': 0}
- Data source license: review TinyStories' CDLA-Sharing-1.0 terms before redistributing any text.

## Model
- Architecture: two-layer stacked LSTM with many-to-many next-token prediction
- Parameters: 4,577,088
- Vocabulary: 8,000 byte-level BPE units
- Sequence length / stride: 128 / 96
- Best epoch: 20
- Training epochs recorded: 20

## Held-out test metrics
- Test loss: 2.4072 nats per subword token
- Test perplexity: 11.10 for this tokenizer
- Test next-token accuracy: 46.41%

These are next-token prediction metrics, not the percentage of complete stories that are correct. Values from different tokenizers should not be compared directly without a correctly normalized metric.

## Artifact pairing
The v3 model and tokenizer are a matched pair. Check hashes in `model_v3.json`. Use raw logits at the final valid prompt position for autoregressive generation.
