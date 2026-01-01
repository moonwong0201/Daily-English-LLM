import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 第4-6行：初始化BERT基础无大小写敏感模型的分词器，指定预训练模型路径
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)
# 第8-13行：初始化掩码语言模型，指定预训练模型路径、数据类型为float16、自动分配设备、注意力实现方式为sdpa
model = AutoModelForMaskedLM.from_pretrained(
    "google-bert/bert-base-uncased",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
# 第15行：对目标句子进行分词处理，返回PyTorch张量格式，并将张量移动到模型所在设备
inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt").to(model.device)
with torch.no_grad():
    # 第19行：将输入传入模型进行前向传播，得到模型输出
    outputs = model(**inputs)
    # 第20行：从模型输出中提取预测的对数概率
    predictions = outputs.logits

# 第23行：查找输入中掩码标记对应的索引位置
mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
# 第24行：根据掩码位置，从预测结果中找出概率最大的标记ID
predicted_token_id = predictions[0, mask_index].argmax(dim=-1)
# 第25行：将预测的标记ID解码为对应的文本 Token
predicted_token = tokenizer.decode(predicted_token_id)

# 第28行：打印最终预测的 Token 结果
print(predicted_token)