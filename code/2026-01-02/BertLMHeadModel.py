from transformers import AutoTokenizer, BertLMHeadModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertLMHeadModel.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])

loss = outputs.loss
logits = outputs.logits

print(loss, logits)
