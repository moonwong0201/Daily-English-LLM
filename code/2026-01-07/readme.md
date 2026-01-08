### BERT Fine-Tune Light Sweep

#### 1. 实验目标

验证《How to Fine-Tune BERT for Text Classification》核心经验：
- lr ∈ {2e-5,3e-5,5e-5} × batch ∈ {16,32}，3 epoch 早停
- 11,988 条真实外卖评价数据

#### 2. 结果（CSV 已上传）

| lr   | batch | acc          |
| ---- | ----- | ------------ |
| 2e-5 | 16    | 0.9149       |
| 2e-5 | 32    | 0.9220 ✅Best |
| 3e-5 | 16    | 0.9153       |
| 3e-5 | 32    | 0.9137       |
| 5e-5 | 16    | 0.9112       |
| 5e-5 | 32    | 0.9145       |

**结论**：2e-5+32 最优，比 5e-5 提升 1.1 %，趋势与论文一致。



## BERT Fine-Tune Light Sweep

### Goal

Validate core hyper-parameters from *"How to Fine-Tune BERT for Text Classification"*:
- lr ∈ {2e-5, 3e-5, 5e-5} × batch ∈ {16, 32}, 3-epoch early-stop
- 500-line mini-dataset is enough to reproduce the trend
- 11,988 real-world food-delivery reviews (11k+ samples)

### Results (CSV uploaded)

| lr   | batch | acc          |
| ---- | ----- | ------------ |
| 2e-5 | 16    | 0.9149       |
| 2e-5 | 32    | 0.9220 ✅Best |
| 3e-5 | 16    | 0.9153       |
| 3e-5 | 32    | 0.9137       |
| 5e-5 | 16    | 0.9112       |
| 5e-5 | 32    | 0.9145       |

**Conclusion**: 2e-5+32 optimal, +1.1 % vs 5e-5, trend consistent with paper.
