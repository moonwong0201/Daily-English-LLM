### BertModel

BERT 的**基础编码器模型**，仅包含 Transformer 编码器和池化层，不包含任务专用头。核心作用是 “提取文本特征”，输出的特征可用于自定义下游任务（如特征拼接、聚类）

forward()输入：

input_ids 文本分词后的token索引

attention_mask 填充掩码（1是有效token，0是填充token）

token_type_ids 句子区分掩码

使用场景：自定义下游任务、文本特征提取

## 任务专用模型类

### BertForPreTraining

核心任务：预训练 MLM+NSP

任务头结构：MLM 头（预测掩码 token）+ NSP 头（二分类）

关键输出：`loss`（总损失）、`prediction_logits`（MLM 预测）、`seq_relationship_logits`（NSP 预测）

### BertLMHeadModel

核心任务：因果语言模型

任务头结构：LM头  将编码器隐藏状态映射到词汇表，输出**左到右预测的词汇得分**

关键输出：`loss`（MLM 损失）、`logits`（token 预测概率）

### BertForMaskedLM

核心任务：掩码语言模型

任务头结构：MLM头  将编码器隐藏状态映射到词汇表，输出**掩码位置的词汇预测得分**

关键输出：

- 推理模式： **`logits`**
- 评估/训练模式： **`logits`**（预测得分）、**`loss`**（损失值）



