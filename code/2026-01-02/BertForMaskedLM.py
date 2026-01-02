from transformers import AutoTokenizer, BertForMaskedLM
import torch

# 第3行：加载指定预训练模型对应的分词器
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# 第4行：加载BERT掩码语言模型预训练模型
model = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
# 第6行：对目标句子进行分词处理，转换为模型可接受的张量格式输入
inputs = tokenizer(
    "The capital of France is [MASK].",
    max_length=20,
    padding="max_length",
    return_tensors="pt",
    truncation=True
)
with torch.no_grad():
    # 第9行：将输入传入模型进行推理，获取词汇预测对数概率输出
    outputs = model(**inputs)
    logits = outputs.logits
# 第12行：查找输入中掩码标记对应的索引位置
mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
# 第14行：根据掩码位置的预测结果，获取概率最大的词汇ID
prediction_token_id = logits[0, mask_index].argmax(dim=-1)
# 第15行：将预测的词汇ID解码为对应的自然语言词汇
prediction_token = tokenizer.decode(prediction_token_id)
# 第17行：对真实句子进行分词处理，获取对应的词汇ID作为标签
labels = tokenizer(
    "The capital of France is Paris.",
    return_tensors="pt",
    max_length=20,
    padding="max_length",
    truncation=True
)["input_ids"]
# 第19行：对标签进行处理，仅保留掩码位置的真实标签，非掩码位置标记为无效值
labels = torch.where(inputs["input_ids"] == tokenizer.mask_token_id, labels, -100)
# 第21行：将输入和标签传入模型，获取包含损失值的输出结果
outputs = model(**inputs, labels=labels)
# 第22行：提取损失值并转换为标量，保留两位小数
loss = round(outputs.loss.item(), 2)

print(f"预测的掩码词汇：{prediction_token}")
print(f"模型损失值：{loss}")
