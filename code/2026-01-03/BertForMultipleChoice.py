from transformers import AutoTokenizer, BertForMultipleChoice
import torch

# 第3步：加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# 第4步：加载多项选择任务的BERT预训练模型
model = BertForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
# 第6步：定义多项选择任务的题干文本
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
# 第7步：定义多项选择任务的第一个选项文本
choice0 = "It is eaten with a fork and a knife."
# 第8步：定义多项选择任务的第二个选项文本
choice1 = "It is eaten while held in the hand."
# 第9步：定义多项选择任务的真实标签并调整张量形状
labels = torch.tensor(0).unsqueeze(0)
# 第11步：对题干和所有选项进行分词编码并做填充处理
encoding = tokenizer(
    [prompt, prompt],
    [choice0, choice1],
    return_tensors="pt",
    padding=True
)
# 第12步：调整编码张量形状后传入模型，并传入真实标签获取输出
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)
# 第15步：提取模型输出的损失值
loss = outputs.loss
# 第16步：提取模型输出的预测得分
logits = outputs.logits
print(loss)
print(logits)
