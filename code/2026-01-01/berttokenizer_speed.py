# day1_berttokenizer_speed.py
from transformers import BertTokenizer, BertTokenizerFast
import time, torch


def tok_speed(text: str):
    # 1. 加载普通 tokenizer 与 Fast tokenizer
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizer_fast = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    # 2. 记录普通版 encode 耗时
    start = time.time()
    ids = tokenizer.encode(text)
    t_normal = time.time() - start
    # 3. 记录 Fast 版 encode 耗时
    start_fast = time.time()
    ids_fast = tokenizer_fast.encode(text)
    t_fast = time.time() - start_fast
    # 4. 返回两者时间比
    ratio = t_fast / t_normal
    return ratio


if __name__ == "__main__":
    txt = "Hello world " * 2500
    ratio = tok_speed(txt)
    print(f"Fast / normal = {ratio:.2f}x")
    if ratio < 1.0:
        print("Fast tokenizer is quicker")
    else:
        print("Fast tokenizer NOT quicker")
    assert ratio < 1.0, "Fast tokenizer should be quicker"