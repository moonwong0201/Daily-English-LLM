import random

import pandas as pd, torch, os
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

DATA_DIR = '/Users/wangyingyue/materials/大模型学习资料——八斗/homework/week4'                       # ① 放 train.tsv & dev.tsv
MODEL_NAME = 'bert-base-chinese'
SEQ_LEN = 128
LR_LIST = [2e-5, 3e-5, 5e-5]
BS_LIST = [16, 32]
EPOCHS = 3
TRAIN_RATIO = 0.8

# ---------- 1. 数据 ----------
class WaimaiDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = dataframe['review'].tolist()
        self.labels = dataframe['label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ---------- 2. 扫点 ----------
def sweep():
    csv_path = os.path.join(DATA_DIR, '作业数据-waimai_10k.csv')
    df = pd.read_csv(csv_path, sep=',')
    df = df.dropna(subset=['label', 'review'])
    df['label'] = df['label'].astype(int)

    total_size = len(df)
    train_size = int(total_size * TRAIN_RATIO)
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    dev_indices = indices[train_size:]

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_df = df.iloc[train_indices]
    dev_df = df.iloc[dev_indices]
    train_ds = WaimaiDataset(train_df, tokenizer)
    dev_ds = WaimaiDataset(dev_df, tokenizer)
    print(f"📊 数据拆分完成：训练集{len(train_ds)}条，验证集{len(dev_ds)}条")

    results = []

    for lr in LR_LIST:
        for bs in BS_LIST:
            args = TrainingArguments(
                output_dir=f'./cls_lr{lr}_bs{bs}',
                learning_rate=lr,
                per_device_train_batch_size=bs,
                num_train_epochs=EPOCHS,
                eval_strategy='epoch',
                save_strategy='epoch',
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model='accuracy',
                logging_steps=50,
                report_to='none',
            )
            model = BertForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=2
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=dev_ds,
                compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(-1) == p.label_ids).mean()}
            )
            trainer.train()
            eval_results = trainer.evaluate()
            acc = trainer.evaluate()['eval_accuracy']
            results.append({'lr': lr, 'bs': bs, 'acc': acc})
            print(f"🔧 完成 lr={lr}, bs={bs} | 验证准确率={acc:.4f}")

    df = pd.DataFrame(results)
    best = df.loc[df['acc'].idxmax()]
    df.to_csv('sweep_results.csv', index=False)
    print(f"✅ Best: lr={best['lr']}, bs={best['bs']}, acc={best['acc']:.4f}")
    print(df)      # 打印全部结果


# ---------- 3. 一键入口 ----------
if __name__ == '__main__':
    sweep()
