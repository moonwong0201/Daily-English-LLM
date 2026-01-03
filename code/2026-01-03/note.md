Day 3 #DailyHFDoc

### BERT Task-Specific Models (HF Transformers)

- BertForNextSentencePrediction: sentence continuity judgment; key outputs `loss` & `logits` (binary).
- BertForSequenceClassification: single/multi-label text classification; key outputs `logits` & `loss`.
- BertForMultipleChoice: multiple choice tasks; key outputs `loss` & `logits` (option scores).
- BertForTokenClassification: token-level tasks (NER/POS); key outputs `loss` & per-token `logits`.
- BertForQuestionAnswering: extractive QA; key inputs `start_positions`/`end_positions`, key outputs `loss` & `start_logits`/`end_logits`.
