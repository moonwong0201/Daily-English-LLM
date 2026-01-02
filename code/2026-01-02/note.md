Day 2 #DailyHFDoc
BERT Model Family (HF v4.56.0)

BertModel = bare encoder + pooler; outputs hidden states.
BertForMaskedLM adds MLM head; key outputs: logits & loss (with labels).
BertForPreTraining combines MLM + NSP heads; total loss = MLM_loss + NSP_loss.
