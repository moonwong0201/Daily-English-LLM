from transformers import AutoTokenizer, BertForTokenClassification
import torch

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# 加载命名实体识别（Token分类）任务的BERT预训练模型
model = BertForTokenClassification.from_pretrained("google-bert/bert-base-uncased")
# 对输入文本进行分词编码，不添加特殊标记并返回PyTorch张量
inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York",
    add_special_tokens=False,
    return_tensors="pt"
)
with torch.no_grad():
    # 模型推理并提取Token分类的预测得分logits
    outputs = model(**inputs)
    logits = outputs.logits
# 在最后一个维度上取最大值，获取每个Token对应的预测类别ID
predicted_token_class_ids = logits.argmax(dim=-1)
# 通过类别ID与标签映射字典，转换得到每个Token对应的预测标签名称
predicted_token_class = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
# 将预测的类别ID作为真实标签赋值
labels = predicted_token_class_ids
# 传入输入和标签，获取模型计算的Token分类损失值
loss = model(**inputs, labels=labels)
# 对损失值进行四舍五入保留两位小数
loss = round(loss.item(), 2)
