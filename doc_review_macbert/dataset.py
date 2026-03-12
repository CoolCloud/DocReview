"""
公文校验数据集类
支持 JSONL 格式的序列标注数据
"""
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from pathlib import Path


# 标签映射
LABEL_MAP = {
    "O": 0,         # 正确字符
    "B-ERROR": 1,   # 错误开始
    "I-ERROR": 2    # 错误内部
}

ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


def load_jsonl(filepath: str) -> List[Dict]:
    """加载 JSONL 格式数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class DocReviewDataset(Dataset):
    """
    公文校验数据集
    
    支持字符级别的序列标注任务
    每个字符标注为 O (正确), B-ERROR (错误开始), I-ERROR (错误延续)
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        """
        初始化数据集
        
        参数:
            data_path: JSONL 数据文件路径
            tokenizer: MacBERT tokenizer
            max_length: 最大序列长度
        """
        self.data = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 统计信息
        error_count = sum(1 for d in self.data if d.get('has_error', False))
        print(f"  ✓ 加载 {len(self.data)} 条数据")
        print(f"  ✓ 含错误: {error_count} 条 ({error_count/len(self.data)*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        返回:
            {
                'input_ids': tensor,       # 输入 token IDs
                'attention_mask': tensor,  # 注意力掩码
                'labels': tensor           # 标签 IDs
            }
        """
        item = self.data[idx]
        text = item['text']
        labels = item['labels']
        
        # MacBERT tokenization
        # use_fast=True 会提供 word_ids 方法用于对齐
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,  # 用于字符级对齐
            return_tensors='pt'
        )
        
        # 对齐标签到 token 级别
        offset_mapping = encoding['offset_mapping'][0]
        label_ids = [-100] * self.max_length  # -100 会被 loss 函数忽略
        
        for i, (start, end) in enumerate(offset_mapping):
            # 跳过特殊 token ([CLS], [SEP], [PAD])
            if start == 0 and end == 0:
                continue
            
            # 将 token 对应到原始字符位置
            # 对于中文，通常一个 token 对应一个字符
            if start < len(labels):
                label_ids[i] = LABEL_MAP.get(labels[start], 0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


def create_data_loaders(
    train_path: str,
    test_path: str,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 0
):
    """
    创建训练和测试数据加载器
    
    参数:
        train_path: 训练集路径
        test_path: 测试集路径
        tokenizer: MacBERT tokenizer
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: 数据加载进程数
    
    返回:
        (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    print("📦 加载数据集...")
    print(f"训练集: {train_path}")
    train_dataset = DocReviewDataset(train_path, tokenizer, max_length)
    
    print(f"\n测试集: {test_path}")
    test_dataset = DocReviewDataset(test_path, tokenizer, max_length)
    
    print(f"\n⚙️  配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  最大长度: {max_length}")
    print()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def print_sample(dataset: DocReviewDataset, idx: int = 0):
    """打印数据样本，用于调试"""
    sample = dataset[idx]
    raw_data = dataset.data[idx]
    
    print("=" * 60)
    print(f"样本 ID: {raw_data['id']}")
    print(f"原文: {raw_data['text']}")
    print(f"含错误: {raw_data['has_error']}")
    print()
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print()
    
    # 解码 tokens
    tokens = dataset.tokenizer.convert_ids_to_tokens(sample['input_ids'])
    labels = sample['labels'].tolist()
    
    print("Token 对齐:")
    for i, (token, label_id) in enumerate(zip(tokens[:50], labels[:50])):
        if label_id != -100:
            label = ID2LABEL.get(label_id, "?")
            print(f"  {i:3d}: {token:10s} -> {label}")
    print("=" * 60)


if __name__ == "__main__":
    # 测试代码
    from transformers import BertTokenizer
    
    print("测试数据集加载...")
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
    
    dataset = DocReviewDataset(
        data_path="data/train.jsonl",
        tokenizer=tokenizer,
        max_length=128
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    print_sample(dataset, 0)
