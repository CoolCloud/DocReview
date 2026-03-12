"""
DistilBERT 情感分析训练脚本
从 Colab notebook 提取并改进
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast
from tqdm import tqdm
import os
import argparse

from model import CustomDistilBertForSequenceClassification
from dataset import create_data_loaders


def get_device():
    """获取可用设备"""
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


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    训练一个 epoch
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        epoch: 当前 epoch 编号
    
    返回:
        平均损失
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for i, batch in enumerate(progress_bar):
        # 数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
        
        # 每 100 个 batch 打印一次
        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch}, Batch {i+1}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    评估模型
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
    
    返回:
        平均损失, 准确率
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total
    
    return avg_loss, accuracy


def train(
    csv_file,
    output_dir='./sentiment_models',
    num_epochs=10,
    batch_size=16,
    learning_rate=5e-5,
    max_length=512,
    train_split=0.8,
    freeze_base=False
):
    """
    完整训练流程
    
    参数:
        csv_file: 数据 CSV 文件路径
        output_dir: 模型保存目录
        num_epochs: 训练轮数
        batch_size: batch 大小
        learning_rate: 学习率
        max_length: 最大序列长度
        train_split: 训练集比例
        freeze_base: 是否冻结 DistilBERT 基础层
    """
    print("="*60)
    print("DistilBERT 情感分析训练")
    print("="*60)
    
    # 设备
    device = get_device()
    
    # 加载 tokenizer
    print("\n加载 tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, test_loader = create_data_loaders(
        csv_file=csv_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        train_split=train_split,
        max_length=max_length
    )
    
    # 创建模型
    print("\n创建模型...")
    model = CustomDistilBertForSequenceClassification(num_labels=3)
    
    # 是否冻结基础层
    if freeze_base:
        model.freeze_base_model()
    
    # 显示参数统计
    params_info = model.get_trainable_params()
    print(f"总参数: {params_info['total']:,}")
    print(f"可训练参数: {params_info['trainable']:,}")
    print(f"冻结参数: {params_info['frozen']:,}")
    
    model.to(device)
    
    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print(f"\n开始训练 ({num_epochs} epochs)...")
    print("="*60)
    
    best_accuracy = 0
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"✓ 保存最佳模型 (准确率: {test_acc:.4f})")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # 保存 tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最佳准确率: {best_accuracy:.4f}")
    print(f"模型已保存到: {output_dir}/")
    
    return best_accuracy


def main():
    parser = argparse.ArgumentParser(description='训练 DistilBERT 情感分析模型')
    parser.add_argument('--csv_file', type=str, required=True, help='训练数据 CSV 文件路径')
    parser.add_argument('--output_dir', type=str, default='./sentiment_models', help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch 大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--train_split', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--freeze_base', action='store_true', help='冻结 DistilBERT 基础层')
    
    args = parser.parse_args()
    
    train(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        train_split=args.train_split,
        freeze_base=args.freeze_base
    )


if __name__ == "__main__":
    main()
