"""
情感分析数据集类
从 Colab notebook 提取并改进
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


class ReviewDataset(Dataset):
    """
    餐厅评论数据集
    支持情感分类任务: negative (0), neutral (1), positive (2)
    """
    
    def __init__(self, csv_file, tokenizer, max_length=512):
        """
        初始化数据集
        
        参数:
            csv_file: CSV 文件路径，包含 'Review' 和 'Rating' 列
            tokenizer: DistilBert tokenizer
            max_length: 最大序列长度
        """
        self.dataset = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 情感映射：文本 -> 数字
        self.label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # 如果 Rating 是文本，转换为数字（支持 object 和 str 类型）
        rating_dtype = str(self.dataset['Rating'].dtype)
        if rating_dtype in ['object', 'str', 'string']:
            # 清理空格并转换为小写
            self.dataset['Rating'] = self.dataset['Rating'].astype(str).str.strip().str.lower()
            self.dataset['Rating'] = self.dataset['Rating'].map(self.label_dict)
            # 删除无法映射的行
            self.dataset = self.dataset.dropna(subset=['Rating'])
            self.dataset['Rating'] = self.dataset['Rating'].astype(int)
            self.dataset = self.dataset.reset_index(drop=True)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回:
            dict: 包含 input_ids, attention_mask, labels
        """
        review_text = str(self.dataset.iloc[idx, 0])  # 第一列：Review 文本
        sentiment = int(self.dataset.iloc[idx, 1])    # 第二列：Rating 标签
        
        # Tokenize 文本
        encoding = self.tokenizer(
            review_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }
    
    @staticmethod
    def load_from_dataframe(df, tokenizer, max_length=512):
        """
        从 DataFrame 创建数据集（不需要 CSV 文件）
        
        参数:
            df: pandas DataFrame，需要有 'Review' 和 'Rating' 列
            tokenizer: tokenizer 实例
            max_length: 最大序列长度
        """
        # 临时保存到 CSV
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            dataset = ReviewDataset(temp_path, tokenizer, max_length)
            return dataset
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def create_data_loaders(csv_file, tokenizer, batch_size=16, train_split=0.8, max_length=512):
    """
    创建训练和测试 DataLoader
    
    参数:
        csv_file: CSV 文件路径
        tokenizer: tokenizer 实例
        batch_size: batch 大小
        train_split: 训练集比例
        max_length: 最大序列长度
    
    返回:
        train_loader, test_loader
    """
    from torch.utils.data import DataLoader, random_split
    
    # 创建完整数据集
    full_dataset = ReviewDataset(csv_file, tokenizer, max_length)
    
    # 划分训练集和测试集
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size
    
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size]
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"数据集大小: {total_size}")
    print(f"训练集: {train_size} 样本 ({len(train_loader)} batches)")
    print(f"测试集: {test_size} 样本 ({len(test_loader)} batches)")
    
    return train_loader, test_loader
