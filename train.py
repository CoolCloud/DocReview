"""
公文核校模型训练脚本
使用 DistilBERT 进行序列标注任务
支持 Mac GPU (MPS) 加速
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import os


# 标签映射
LABEL_MAP = {"O": 0, "B-ERROR": 1, "I-ERROR": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


class DocReviewDataset(Dataset):
    """公文核校数据集"""
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 对齐标签
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = [-100] * self.max_length
        
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(labels):
                label_ids[i] = LABEL_MAP.get(labels[word_id], 0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


def get_device():
    """获取最佳设备（优先使用 Mac GPU）"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ 使用 Mac GPU (MPS) 加速训练")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ 使用 CUDA GPU 加速训练")
    else:
        device = torch.device("cpu")
        print("⚠ 使用 CPU 训练（速度较慢）")
    return device


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=-1)
            
            # 收集预测和真实标签
            for pred, label, mask in zip(preds, labels, attention_mask):
                pred_list = pred[mask.bool()].cpu().numpy()
                label_list = label[mask.bool()].cpu().numpy()
                
                # 过滤掉填充标签
                valid_indices = label_list != -100
                predictions.extend(pred_list[valid_indices])
                true_labels.extend(label_list[valid_indices])
    
    # 计算指标
    print("\n分类报告:")
    
    # 获取实际存在的标签
    unique_labels = sorted(list(set(true_labels) | set(predictions)))
    label_names = [ID2LABEL.get(l, f"LABEL_{l}") for l in unique_labels]
    
    print(classification_report(
        true_labels, 
        predictions,
        labels=unique_labels,
        target_names=label_names,
        digits=4,
        zero_division=0
    ))
    
    # 计算准确率
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    return accuracy


def main():
    """主训练流程"""
    print("="*60)
    print("公文核校模型训练")
    print("="*60)
    
    # 超参数
    MODEL_NAME = "bert-base-chinese"  # 使用中文 BERT
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    
    # 设备
    device = get_device()
    
    # 加载 tokenizer 和模型
    print(f"\n加载模型: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP)
    )
    model.to(device)
    
    # 加载数据
    print("\n加载数据集...")
    train_dataset = DocReviewDataset("data/train.json", tokenizer, MAX_LENGTH)
    test_dataset = DocReviewDataset("data/test.json", tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"测试样本: {len(test_dataset)}")
    
    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # 训练循环
    print(f"\n开始训练 ({EPOCHS} epochs)...")
    print("="*60)
    
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 60)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"训练损失: {train_loss:.4f}")
        
        # 评估
        accuracy = evaluate(model, test_loader, device)
        print(f"测试准确率: {accuracy:.4f}")
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs("models", exist_ok=True)
            model.save_pretrained("models/best_model")
            tokenizer.save_pretrained("models/best_model")
            print(f"✓ 保存最佳模型 (准确率: {accuracy:.4f})")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最佳准确率: {best_accuracy:.4f}")
    print(f"模型已保存到: models/best_model/")


if __name__ == "__main__":
    main()
