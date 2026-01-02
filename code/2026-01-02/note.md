Day 2 #DailyHFDoc

### BERT Model Family (HF v4.56.0)

- BertModel = bare encoder + pooler; outputs hidden states for text feature extraction.
- BertForPreTraining combines MLM + NSP heads; total loss = MLM_loss + NSP_loss; outputs prediction_logits & seq_relationship_logits.
- BertLMHeadModel adds LM head; key outputs: logits & CLM loss; for left-to-right token prediction.
- BertForMaskedLM adds MLM head; key outputs: logits (inference) & logits + loss (with labels).
