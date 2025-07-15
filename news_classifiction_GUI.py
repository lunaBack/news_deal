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
import tkinter as tk
from tkinter import messagebox, scrolledtext

# ============ 全局参数 ============
LABEL_MAP = {"娱乐": 0, "体育": 1, "教育": 2, "时政": 3, "科技": 4,
             "房产": 5, "社会": 6, "股票": 7, "财经": 8, "家居": 9,
             "游戏": 10, "时尚": 11, "彩票": 12, "星座": 13}
NUM_CLASSES = len(LABEL_MAP)

MAX_SEQ_LEN = 128
BATCH_SIZE = 32

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

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

# ============ 创建数据加载器 ============
def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda samples: (
            Pad(axis=0, pad_val=dataset.tokenizer.pad_token_id)([s['input_ids'] for s in samples]),
            Pad(axis=0, pad_val=0)([s['token_type_ids'] for s in samples]),
            Pad(axis=0, pad_val=0)([s['attention_mask'] for s in samples]),
            Stack()([s['labels'] for s in samples])
        ),
        num_workers=4  # 设置多线程加载
    )

# ============ GUI界面 ============
class NewsClassificationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BERT新闻分类系统")

        self.tokenizer = None
        self.model = None

        self.create_widgets()

    def create_widgets(self):
        # 创建菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 添加菜单项
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="测试集预测", command=self.test_prediction)
        file_menu.add_command(label="用户输入预测", command=self.user_prediction)
        file_menu.add_command(label="重新训练", command=self.retrain_model)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

        # 创建文本输入区域
        self.text_frame = tk.LabelFrame(self.root, text="新闻内容")
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.text_widget = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD)
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建按钮区域
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.predict_button = tk.Button(self.button_frame, text="预测分类", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_button = tk.Button(self.button_frame, text="清空内容", command=self.clear_text)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)

        # 创建结果显示区域
        self.result_frame = tk.LabelFrame(self.root, text="预测结果")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_text = scrolledtext.ScrolledText(self.result_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 初始化模型和分词器
        self.init_model_and_tokenizer()

    def init_model_and_tokenizer(self):
        # 初始化分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-wwm-ext-chinese')

        # 加载模型
        self.model = load_bert_model()

        # 加载训练好的模型参数
        if os.path.exists("best_model.pdparams"):
            state_dict = paddle.load("best_model.pdparams")
            self.model.set_state_dict(state_dict)
            print("加载模型参数成功！")
        else:
            messagebox.showwarning("警告", "未找到训练好的模型文件，请先训练模型或确保模型文件存在。")

        device = paddle.get_device()
        self.model = self.model.to(device)

    def test_prediction(self):
        # 这里可以实现从测试集中选择数据进行预测的功能
        messagebox.showinfo("测试集预测", "请选择测试集数据进行预测")

    def user_prediction(self):
        # 这里可以实现用户输入数据进行预测的功能
        messagebox.showinfo("用户输入预测", "请在文本框中输入新闻内容进行预测")

    def show_about(self):
        about_text = "BERT新闻分类系统\n版本 1.0\n使用PaddlePaddle和PaddleNLP框架实现"
        messagebox.showinfo("关于", about_text)

    def retrain_model(self):
        # 实现重新训练模型的功能
        # 这里可以调用训练函数重新训练模型
        # 确保在调用此功能前已经加载了数据集
        messagebox.showinfo("重新训练", "重新训练功能尚未实现")

    def predict_news(self, text):
        inputs = self.tokenizer(text, max_length=MAX_SEQ_LEN, padding='max_length', truncation=True, return_attention_mask=True, return_token_type_ids=True)
        input_ids = paddle.to_tensor([inputs["input_ids"]], dtype='int64')
        token_type_ids = paddle.to_tensor([inputs["token_type_ids"]], dtype='int64')
        attention_mask = paddle.to_tensor([inputs["attention_mask"]], dtype='int64')

        device = paddle.get_device()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        with paddle.no_grad():
            logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, axis=1).numpy()  # 获取预测概率

        return probs

    def predict(self):
        text = self.text_widget.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "请输入新闻内容！")
            return

        probs = self.predict_news(text)
        max_prob_index = np.argmax(probs)  # 获取最大概率的索引
        final_result = REVERSE_LABEL_MAP[max_prob_index]

        results = "所有类别的预测概率：\n"
        results += "\n".join([f"{REVERSE_LABEL_MAP[i]}: {prob:.4f}" for i, prob in enumerate(probs[0])])
        results += f"\n\n最终预测类别: {final_result}"

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, results)

    def clear_text(self):
        self.text_widget.delete("1.0", tk.END)
        self.result_text.delete("1.0", tk.END)

# ============ 主程序 ============
def main():
    paddle.seed(2026)
    device = paddle.get_device()
    if 'gpu' in device:
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    print(f"使用设备: {device}")

    # 启动GUI
    root = tk.Tk()
    app = NewsClassificationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()