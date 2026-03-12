"""
自定义 DistilBERT 情感分类模型
从 Colab notebook 提取并改进
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel


class CustomDistilBertForSequenceClassification(nn.Module):
    """
    自定义 DistilBERT 序列分类模型
    
    架构:
        DistilBERT base → Pre-classifier (Linear + ReLU) → Dropout → Classifier (Linear)
    """
    
    def __init__(self, num_labels=3, dropout_prob=0.3, pretrained_model='distilbert-base-uncased'):
        """
        初始化模型
        
        参数:
            num_labels: 分类类别数（默认 3：negative/neutral/positive）
            dropout_prob: Dropout 概率
            pretrained_model: 预训练模型名称
        """
        super(CustomDistilBertForSequenceClassification, self).__init__()
        
        # 加载预训练 DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)
        
        # 分类头
        hidden_size = 768  # DistilBERT 隐藏层维度
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        参数:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        返回:
            logits: 分类 logits [batch_size, num_labels]
        """
        # DistilBERT 编码
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # hidden_state: [batch_size, seq_len, hidden_size]
        hidden_state = distilbert_output[0]
        
        # 使用 [CLS] token 的表示（第一个 token）
        pooled_output = hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # 通过分类头
        pooled_output = self.pre_classifier(pooled_output)  # [batch_size, hidden_size]
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        return logits
    
    def freeze_base_model(self):
        """
        冻结 DistilBERT 基础层，只训练分类头
        适用于小数据集或快速训练
        """
        for param in self.distilbert.parameters():
            param.requires_grad = False
        print("✓ DistilBERT 基础层已冻结")
    
    def unfreeze_base_model(self):
        """
        解冻 DistilBERT 基础层，进行全模型微调
        """
        for param in self.distilbert.parameters():
            param.requires_grad = True
        print("✓ DistilBERT 基础层已解冻")
    
    def get_trainable_params(self):
        """
        获取可训练参数数量
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


def load_model(model_path, num_labels=3, device='cpu'):
    """
    加载保存的模型
    
    参数:
        model_path: 模型文件路径
        num_labels: 类别数
        device: 设备
    
    返回:
        加载的模型
    """
    model = CustomDistilBertForSequenceClassification(num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_model(model, save_path):
    """
    保存模型
    
    参数:
        model: 模型实例
        save_path: 保存路径
    """
    torch.save(model.state_dict(), save_path)
    print(f"✓ 模型已保存到: {save_path}")
