import torch
from transformers import AutoTokenizer, BertForSequenceClassification

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# 加载默认二分类BERT分类模型
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

# 对输入文本进行分词编码处理
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# 模型推理（无梯度计算）
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    print("类别0和类别1的原始得分：", logits)
    # 获取预测类别ID
    predicted_class_id = logits.argmax().item()
    # 获取标签名称（默认映射）
    label_name = model.config.id2label[predicted_class_id]
    print(f"预测类别ID：{predicted_class_id}，预测标签名称：{label_name}")

# 正确获取分类类别数量
num_labels = model.num_labels
print(f"真实分类类别数：{num_labels}")

# 定义真实标签（二分类下，标签值为0或1）
labels = torch.tensor([1])
# 传入输入和标签，计算损失
outputs_with_loss = model(**inputs, labels=labels)
loss = outputs_with_loss.loss
# 损失值格式化
loss = round(loss.item(), 2)
print(f"模型损失值：{loss}")