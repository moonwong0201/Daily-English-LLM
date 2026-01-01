import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="google-bert/bert-base-uncased",
    dtype=torch.float16,
    device=0
)
result = pipeline("Plants create [MASK] through a process known as photosynthesis.")
print(result)
