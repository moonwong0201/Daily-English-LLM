Day 5 BERT: Pre-training of Deep Bidirectional Transformers

## BERT
1. Two pre-training strategies: feature-based (frozen) vs. fine-tuning (updatable).  
2. BERT uses Masked Language Model (MLM) + Next Sentence Prediction (NSP) to enable deep bidirectional representations.  
3. Achieves SOTA on 11 NLP tasks with minimal task-specific architecture.  

Next: 30-line PyTorch core implementation verifies embedding → Transformer → MLM head pipeline.
