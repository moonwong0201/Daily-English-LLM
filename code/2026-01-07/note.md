Day7 How to Fine-Tune BERT for Text Classification?

### General Fine-tuning Scheme for BERT Model

- Further pre-training (task-specific / domain-specific)
- Optional multi-task fine-tuning
- Target task fine-tuning
- Combined with optimizations such as **head-tail truncation** for long texts and **layer-wise decreasing learning rate**

### Key Findings

- Domain-specific pre-training outperforms task-specific pre-training
- The top-layer features of BERT are most suitable for classification tasks
- Low learning rates can alleviate catastrophic forgetting
- Further pre-training yields significant gains in few-shot scenarios
