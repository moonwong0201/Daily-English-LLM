from transformers import AutoTokenizer, BertForQuestionAnswering
import torch

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# 加载问答任务的BERT预训练模型
model = BertForQuestionAnswering.from_pretrained("google-bert/bert-base-uncased")
# 定义问答任务的问题和上下文文本
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
# 对问题和上下文文本进行分词编码，返回PyTorch张量
inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    # 模型推理并获取问答任务的输出对象
    outputs = model(**inputs)
# 从起始位置预测得分中取最大值，得到答案起始索引
answer_start_index = outputs.start_logits.argmax(dim=-1)
# 从结束位置预测得分中取最大值，得到答案结束索引
answer_end_index = outputs.end_logits.argmax(dim=-1)
# 根据起始和结束索引，从输入的token索引中截取答案对应的token张量
predicted_answer_token = inputs["input_ids"][0, answer_start_index:answer_end_index + 1]
# 对答案token张量进行解码，跳过特殊标记，得到人类可理解的答案文本
answer = tokenizer.decode(predicted_answer_token, skip_special_tokens=True)
# 定义真实答案的起始索引张量
target_start_index = torch.tensor([14])
# 定义真实答案的结束索引张量
target_end_index = torch.tensor([15])
# 传入输入、真实答案起始索引和结束索引，获取模型输出
outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
# 提取模型输出的损失值
loss = outputs.loss
# 对损失值进行四舍五入保留两位小数
loss = round(loss.item(), 2)
