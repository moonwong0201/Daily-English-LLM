from transformers import AutoTokenizer, BertForNextSentencePrediction
import torch

# 第3步：加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# 第4步：加载下一句预测任务的BERT预训练模型
model = BertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")
# 第6步：定义第一句提示文本
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
# 第7步：定义待判断的下一句文本
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
# 第8步：对两句文本进行分词编码处理
encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
# 第10步：将编码输入模型并传入标签，获取输出结果
outputs = model(**encoding, labels=torch.LongTensor([1]))  # 1表示这组句子对是 “随机无关” 的
# 第11步：提取模型输出的预测对数概率
logits = outputs.logits
# 第12步：断言验证下一句为随机句子的预测结果
assert logits[0, 0] < logits[0, 1]
print(logits[0, 0])
print(logits[0, 1])