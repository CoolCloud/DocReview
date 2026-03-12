"""
基于 MacBERT 的公文校验模型

MacBERT (Mac as correction BERT) 是针对中文进行改进的 BERT 模型：
- 使用 MLM (Masked Language Modeling) 作为 correction
- 使用同义词替换而不是 [MASK]
- 对中文文本任务表现更好
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from typing import Optional, Tuple


class MacBERTForDocReview(BertPreTrainedModel):
    """
    基于 MacBERT 的序列标注模型
    用于检测公文中的错误位置
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # MacBERT 编码器
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Dropout 层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 分类头
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        前向传播
        
        参数:
            input_ids: 输入 token IDs, shape: (batch_size, seq_length)
            attention_mask: 注意力掩码, shape: (batch_size, seq_length)
            token_type_ids: token 类型 IDs (可选)
            labels: 标签, shape: (batch_size, seq_length)
            return_dict: 是否返回字典格式
        
        返回:
            如果 return_dict=True:
                {
                    'loss': tensor,        # 损失值 (如果提供 labels)
                    'logits': tensor,      # 预测 logits, shape: (batch_size, seq_length, num_labels)
                    'hidden_states': ...,  # 隐藏状态 (如果 config.output_hidden_states=True)
                    'attentions': ...      # 注意力权重 (如果 config.output_attentions=True)
                }
            否则返回 tuple
        """
        # BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # 获取序列输出
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # 分类
        logits = self.classifier(sequence_output)  # (batch_size, seq_length, num_labels)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 只计算非 -100 标签的损失
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        预测接口（推理时使用）
        
        参数:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
        
        返回:
            predicted_labels: 预测的标签 IDs, shape: (batch_size, seq_length)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
        return predictions


def create_macbert_model(
    model_name: str = "hfl/chinese-macbert-base",
    num_labels: int = 3,
    from_pretrained: bool = True
) -> MacBERTForDocReview:
    """
    创建 MacBERT 模型
    
    参数:
        model_name: 预训练模型名称
            - 'hfl/chinese-macbert-base': 基础版 (102M 参数)
            - 'hfl/chinese-macbert-large': 大版本 (324M 参数)
        num_labels: 标签数量 (默认 3: O, B-ERROR, I-ERROR)
        from_pretrained: 是否从预训练模型加载
    
    返回:
        MacBERTForDocReview 模型
    """
    from transformers import BertConfig
    
    if from_pretrained:
        from transformers import BertTokenizer
        print(f"📥 加载预训练模型: {model_name}")
        
        # 加载配置
        config = BertConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        
        # 加载模型
        model = MacBERTForDocReview.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True  # 分类头大小不匹配时忽略
        )
        
        print(f"✓ 模型加载完成")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    else:
        # 从头训练（不推荐）
        config = BertConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        model = MacBERTForDocReview(config)
        print("⚠️  从随机初始化创建模型（不推荐用于实际任务）")
    
    return model


def print_model_info(model: MacBERTForDocReview):
    """打印模型信息"""
    print("\n" + "=" * 60)
    print("📊 模型信息")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {total_params - trainable_params:,}")
    print(f"标签数量: {model.num_labels}")
    print()
    
    print("模型结构:")
    print(f"  BERT 层: {model.bert.config.num_hidden_layers} 层")
    print(f"  隐藏维度: {model.bert.config.hidden_size}")
    print(f"  注意力头: {model.bert.config.num_attention_heads}")
    print(f"  中间层维度: {model.bert.config.intermediate_size}")
    print(f"  词表大小: {model.bert.config.vocab_size}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 测试代码
    print("测试 MacBERT 模型...")
    
    model = create_macbert_model(
        model_name="hfl/chinese-macbert-base",
        num_labels=3
    )
    
    print_model_info(model)
    
    # 测试前向传播
    batch_size = 2
    seq_length = 32
    
    input_ids = torch.randint(0, 21128, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 3, (batch_size, seq_length))
    
    print("测试前向传播...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"✓ Loss: {outputs['loss'].item():.4f}")
    print(f"✓ Logits shape: {outputs['logits'].shape}")
    print("\n模型测试完成！")
