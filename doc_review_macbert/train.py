"""
MacBERT 公文校验训练脚本
支持 Mac GPU (MPS), CUDA 和 CPU
"""
import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from pathlib import Path
import argparse
from datetime import datetime

# 支持模块导入和直接运行
try:
    from .model import create_macbert_model, print_model_info
    from .dataset import create_data_loaders, LABEL_MAP, ID2LABEL
except ImportError:
    from model import create_macbert_model, print_model_info
    from dataset import create_data_loaders, LABEL_MAP, ID2LABEL


def get_device():
    """获取可用的最佳设备"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ 使用 Mac GPU (MPS) 加速训练")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 使用 CUDA GPU 加速训练 ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("⚠️  使用 CPU 训练（速度较慢）")
    return device


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """
    训练一个 epoch
    
    参数:
        model: MacBERT 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        epoch: 当前 epoch 编号
    
    返回:
        平均损失
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)
    
    for batch in progress_bar:
        # 数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        
        # 计算准确率（只计算非 -100 的位置）
        predictions = torch.argmax(logits, dim=-1)
        mask = (labels != -100)
        correct += ((predictions == labels) & mask).sum().item()
        total += mask.sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc


def evaluate(model, test_loader, device):
    """
    评估模型
    
    参数:
        model: MacBERT 模型
        test_loader: 测试数据加载器
        device: 设备
    
    返回:
        (avg_loss, accuracy, f1, precision, recall, report)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中", ncols=100):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # 收集预测和标签
            predictions = torch.argmax(logits, dim=-1)
            
            # 只保留有效位置（非 -100）
            mask = (labels != -100)
            valid_predictions = predictions[mask].cpu().numpy()
            valid_labels = labels[mask].cpu().numpy()
            
            all_predictions.extend(valid_predictions)
            all_labels.extend(valid_labels)
    
    avg_loss = total_loss / len(test_loader)
    
    # 计算指标
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='weighted',
        zero_division=0
    )
    
    # 详细报告
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=target_names,
        zero_division=0
    )
    
    return avg_loss, accuracy, f1, precision, recall, report


def save_model(model, tokenizer, save_dir, metrics=None):
    """保存模型和 tokenizer"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # 保存训练指标
    if metrics:
        metrics_file = save_path / "training_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 模型已保存到: {save_path.absolute()}")


def train(
    train_data_path: str,
    test_data_path: str,
    model_name: str = "hfl/chinese-macbert-base",
    output_dir: str = "models",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 256,
    seed: int = 42
):
    """
    训练 MacBERT 公文校验模型
    
    参数:
        train_data_path: 训练数据路径 (JSONL)
        test_data_path: 测试数据路径 (JSONL)
        model_name: 预训练模型名称
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        warmup_ratio: 预热比例
        max_length: 最大序列长度
        seed: 随机种子
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("\n" + "=" * 60)
    print("🚀 MacBERT 公文校验模型训练")
    print("=" * 60)
    print(f"预训练模型: {model_name}")
    print(f"训练数据: {train_data_path}")
    print(f"测试数据: {test_data_path}")
    print(f"输出目录: {output_dir}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    print("=" * 60 + "\n")
    
    # 获取设备
    device = get_device()
    print()
    
    # 加载 tokenizer
    print("📥 加载 tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(f"✓ 词表大小: {len(tokenizer)}\n")
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        train_path=train_data_path,
        test_path=test_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # 创建模型
    print("🔧 创建模型...")
    model = create_macbert_model(
        model_name=model_name,
        num_labels=len(LABEL_MAP)
    )
    print_model_info(model)
    
    model = model.to(device)
    
    # 优化器和调度器
    print("⚙️  配置优化器...")
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"✓ 总训练步数: {total_steps}")
    print(f"✓ 预热步数: {warmup_steps}\n")
    
    # 训练循环
    print("🏋️  开始训练...\n")
    best_f1 = 0
    training_history = []
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print('='*60)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        
        print(f"\n训练结果: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
        
        # 评估
        print("\n开始评估...")
        test_loss, test_acc, f1, precision, recall, report = evaluate(
            model, test_loader, device
        )
        
        print(f"\n评估结果:")
        print(f"  Loss:      {test_loss:.4f}")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print("\n详细报告:")
        print(report)
        
        # 记录历史
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'timestamp': datetime.now().isoformat()
        }
        training_history.append(epoch_metrics)
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            print(f"\n🎉 新的最佳 F1: {f1:.4f}")
            best_model_dir = Path(output_dir) / "best_model"
            save_model(model, tokenizer, best_model_dir, epoch_metrics)
        
        # 保存检查点
        checkpoint_dir = Path(output_dir) / f"checkpoint_epoch_{epoch}"
        save_model(model, tokenizer, checkpoint_dir, epoch_metrics)
    
    # 保存训练历史
    history_file = Path(output_dir) / "training_history.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"最佳 F1 分数: {best_f1:.4f}")
    print(f"模型保存位置: {Path(output_dir).absolute()}")
    print(f"训练历史: {history_file.absolute()}")
    print("=" * 60 + "\n")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="训练 MacBERT 公文校验模型")
    
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train.jsonl",
        help="训练数据路径 (JSONL)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/test.jsonl",
        help="测试数据路径 (JSONL)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="hfl/chinese-macbert-base",
        choices=["hfl/chinese-macbert-base", "hfl/chinese-macbert-large"],
        help="预训练模型名称"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="输出目录"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批次大小"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="学习率"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="最大序列长度"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 开始训练
    train(
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
