"""
公文核校模型训练脚本 V2
支持 JSONL 格式数据
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
from sklearn.metrics import classification_report, precision_recall_fscore_support
import os
from pathlib import Path


# 标签映射
LABEL_MAP = {"O": 0, "B-ERROR": 1, "I-ERROR": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_jsonl(filepath: str) -> list:
    """加载 JSONL 格式数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class DocReviewDataset(Dataset):
    """公文核校数据集（支持 JSONL 格式）"""
    def __init__(self, data_path, tokenizer, max_length=128):
        # 自动检测文件格式
        if data_path.endswith('.jsonl'):
            self.data = load_jsonl(data_path)
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"  加载 {len(self.data)} 条数据")
        error_count = sum(1 for d in self.data if d.get('has_error', False))
        print(f"  含错误样本: {error_count} ({error_count/len(self.data)*100:.1f}%)")
    
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


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - Training")
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
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
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算指标
    print("\n" + "="*60)
    print("分类报告:")
    print("="*60)
    
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
    
    # 计算错误检测的精确率和召回率（重点关注）
    if 1 in unique_labels:  # B-ERROR 存在
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, 
            predictions, 
            labels=[1, 2],  # B-ERROR 和 I-ERROR
            average='micro',
            zero_division=0
        )
        print(f"错误检测 (B-ERROR + I-ERROR):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    # 计算准确率
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    
    return accuracy, avg_loss, f1 if 1 in unique_labels else 0.0


def main():
    """主训练流程"""
    print("="*70)
    print("公文核校模型训练 V2")
    print("="*70)
    
    # 超参数
    MODEL_NAME = "bert-base-chinese"  # 使用中文 BERT
    BATCH_SIZE = 16  # 增加批次大小
    EPOCHS = 8  # 增加训练轮次
    LEARNING_RATE = 3e-5
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
    print("训练集:")
    train_dataset = DocReviewDataset("data/train.jsonl", tokenizer, MAX_LENGTH)
    print("测试集:")
    test_dataset = DocReviewDataset("data/test.jsonl", tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0  # Mac 上设为 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # 训练循环
    print(f"\n{'='*70}")
    print(f"开始训练")
    print(f"{'='*70}")
    print(f"总 Epochs: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"总训练步数: {total_steps}")
    print(f"{'='*70}\n")
    
    best_f1 = 0
    best_accuracy = 0
    history = []
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*70}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1)
        print(f"\n训练损失: {train_loss:.4f}")
        
        # 评估
        accuracy, eval_loss, f1 = evaluate(model, test_loader, device)
        print(f"验证损失: {eval_loss:.4f}")
        print(f"测试准确率: {accuracy:.4f}")
        
        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'accuracy': accuracy,
            'f1': f1
        })
        
        # 保存最佳模型（基于 F1 分数）
        if f1 > best_f1:
            best_f1 = f1
            best_accuracy = accuracy
            os.makedirs("models", exist_ok=True)
            model.save_pretrained("models/best_model_v2")
            tokenizer.save_pretrained("models/best_model_v2")
            print(f"\n✓ 保存最佳模型 (F1: {f1:.4f}, 准确率: {accuracy:.4f})")
    
    # 保存训练历史
    with open("models/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"最佳 F1 分数: {best_f1:.4f}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    print(f"模型已保存到: models/best_model_v2/")
    print(f"训练历史已保存到: models/training_history.json")
    print("="*70)


if __name__ == "__main__":
    main()
