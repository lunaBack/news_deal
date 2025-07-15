# -*- coding: utf-8 -*-
# 实验：使用BERT进行新闻分类
import os
import random
import time
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.data import Pad, Stack, Tuple
from tqdm import tqdm  # 引入tqdm
import matplotlib.pyplot as plt  # 引入matplotlib

# ============ 全局参数 ============
LABEL_MAP = {"娱乐": 0, "体育": 1, "教育": 2, "时政": 3, "科技": 4,
             "房产": 5, "社会": 6, "股票": 7, "财经": 8, "家居": 9,
             "游戏": 10, "时尚": 11, "彩票": 12, "星座": 13}
NUM_CLASSES = len(LABEL_MAP)

MAX_SEQ_LEN = 16
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
EPOCHS = 1  # 设置为1轮

# ============ 数据加载与预处理 ============
class NewsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.vocab_size = tokenizer.vocab_size  # 获取词汇表大小

        with open(data_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    text, label = parts[0], parts[1]
                    if label in LABEL_MAP:
                        self.data.append({'text': text, 'label': LABEL_MAP[label]})

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example["text"]
        label = example["label"]

        # 使用tokenizer处理文本
        tokenized = self.tokenizer(
            text=text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,

            return_token_type_ids=True
        )

        input_ids = tokenized["input_ids"]
        token_type_ids = tokenized["token_type_ids"]
        attention_mask = tokenized["attention_mask"]

        # 检查input_ids是否有异常值
        if any(id_val < 0 or id_val >= self.vocab_size for id_val in input_ids):
            print(f"警告：发现异常input_ids值！索引：{idx}，文本：{text}，input_ids：{input_ids}")

        # 确保 input_ids 在合法范围内且非负
        input_ids = [max(min(id_val, self.vocab_size - 1), 0) for id_val in input_ids]

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': [label]
        }

    def __len__(self):
        return len(self.data)

# ============ 模型定义 ============
def load_bert_model():
    model = BertForSequenceClassification.from_pretrained(
        'bert-wwm-ext-chinese', num_classes=NUM_CLASSES)
    return model

# ============ 模型训练与评估 ============
def train_model(model, train_loader, valid_loader, lr=LEARNING_RATE):
    # 将模型放置在GPU上（如果可用）
    device = paddle.get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters())

    best_acc = 0.0
    batch_losses = []  # 记录每个batch的训练损失
    valid_accs = []    # 记录验证准确率

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        epoch_loss = 0
        start_time = time.time()

        # 使用tqdm包裹train_loader，添加进度条
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1} Training', dynamic_ncols=True)
        for batch in train_progress:
            # 使用 clone().detach() 替代 paddle.to_tensor()
            input_ids = paddle.to_tensor(batch[0], dtype='int64').clone().detach()
            token_type_ids = paddle.to_tensor(batch[1], dtype='int64').clone().detach()
            attention_mask = paddle.to_tensor(batch[2], dtype='int64').clone().detach()
            labels = paddle.to_tensor(batch[3], dtype='int64').clone().detach()

            # 将数据移动到GPU上（如果可用）
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with paddle.amp.auto_cast():  # 启用混合精度
                logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels.squeeze())

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            epoch_loss += loss.numpy().item()
            batch_losses.append(loss.numpy().item())  # 记录每个batch的损失

            # 更新进度条描述信息
            train_progress.set_postfix({'loss': f'{loss.numpy().item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)

        # 验证阶段
        model.eval()
        correct, total = 0, 0
        # 使用tqdm包裹valid_loader，添加进度条
        valid_progress = tqdm(valid_loader, desc=f'Epoch {epoch + 1} Validation', dynamic_ncols=True)
        for batch in valid_progress:
            # 使用 clone().detach() 替代 paddle.to_tensor()
            input_ids = paddle.to_tensor(batch[0], dtype='int64').clone().detach()
            token_type_ids = paddle.to_tensor(batch[1], dtype='int64').clone().detach()
            attention_mask = paddle.to_tensor(batch[2], dtype='int64').clone().detach()
            labels = paddle.to_tensor(batch[3], dtype='int64').clone().detach()

            # 将数据移动到GPU上（如果可用）
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            preds = paddle.argmax(logits, axis=1)
            correct += (preds == labels.squeeze()).sum().numpy()
            total += len(labels)

            # 更新进度条描述信息
            valid_progress.set_postfix({'acc': f'{correct / total:.4f}'})

        val_acc = correct / total
        valid_accs.append(val_acc)  # 记录验证准确率
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {val_acc:.4f} | Time: {epoch_time:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            paddle.save(model.state_dict(), "best_model.pdparams")

    print(f"训练完成! 最佳验证准确率: {best_acc:.4f}")

    # 绘制单个epoch的训练损失变化和验证准确率图表
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(batch_losses, label='Training Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(['Validation Accuracy'], [valid_accs[0]], color='orange')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

# ============ 创建数据加载器 ============
def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda samples: (
            Pad(axis=0, pad_val=tokenizer.pad_token_id)([s['input_ids'] for s in samples]),
            Pad(axis=0, pad_val=0)([s['token_type_ids'] for s in samples]),
            Pad(axis=0, pad_val=0)([s['attention_mask'] for s in samples]),
            Stack()([s['labels'] for s in samples])
        ),
        num_workers=4  # 设置多线程加载
    )

# ============ 主程序 ============
def main():
    global tokenizer

    print("=" * 70)
    print("实验: BERT新闻分类系统")
    print(f"标签类别: {list(LABEL_MAP.keys())}")
    print("=" * 70)

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('bert-wwm-ext-chinese')
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # 1. 数据加载
    train_dataset = NewsDataset("train.txt", tokenizer, MAX_SEQ_LEN)
    valid_dataset = NewsDataset("valid.txt", tokenizer, MAX_SEQ_LEN)
    test_dataset = NewsDataset("test.txt", tokenizer, MAX_SEQ_LEN)

    # 创建DataLoader
    train_loader = create_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(test_dataset, BATCH_SIZE, shuffle=False)

    print(f"数据加载完成! 训练集样本数: {len(train_dataset)} | 测试集样本数: {len(test_dataset)}")

    # 2. 模型训练与评估
    bert_model = load_bert_model()
    train_model(bert_model, train_loader, valid_loader)

if __name__ == "__main__":
    paddle.seed(2025)
    device = paddle.get_device()
    if 'gpu' in device:
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    print(f"使用设备: {device}")
    main()