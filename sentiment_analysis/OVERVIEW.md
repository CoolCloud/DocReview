# 情感分析模块说明

## 📌 概述

这是从 Google Colab notebook 提取并改进的独立 **DistilBERT 情感分析**模块。

**原始 Notebook**: https://colab.research.google.com/drive/1B_ERSgQDLNOL8NPCvkn7s8h_vEdZ_szI

## 🎯 功能

对英文文本（如餐厅评论）进行三分类情感分析：
- **Negative**: 负面情感
- **Neutral**: 中性情感  
- **Positive**: 正面情感

## 📂 文件说明

| 文件 | 说明 |
|------|------|
| `dataset.py` | 数据集类 `ReviewDataset`，处理 CSV 数据 |
| `model.py` | 自定义 DistilBERT 模型 `CustomDistilBertForSequenceClassification` |
| `train.py` | 完整训练流程，支持命令行参数 |
| `predict.py` | 预测脚本，支持单条/批量/交互式预测 |
| `demo.py` | 使用示例和演示代码 |
| `__init__.py` | 模块初始化，方便导入 |
| `requirements.txt` | 依赖清单 |
| `README.md` | 详细文档 |

## 🚀 使用流程

### 1. 准备数据

CSV 格式，包含 `Review` 和 `Rating` 列：

```csv
Review,Rating
"The food was amazing!",positive
"Terrible service",negative
"It was okay",neutral
```

**下载示例数据**:
```bash
wget https://raw.githubusercontent.com/kyuz0/llm-chronicles/main/datasets/restaurant_reviews.csv
```

### 2. 训练模型

```bash
cd sentiment_analysis
python train.py --csv_file restaurant_reviews.csv --epochs 10 --batch_size 16
```

输出示例：
```
✓ 使用 Mac GPU (MPS) 加速训练
数据集大小: 9951
训练集: 7960 样本 (498 batches)
测试集: 1991 样本 (125 batches)

Epoch 1/10
训练损失: 0.6793, 训练准确率: 0.7245
测试损失: 0.5824, 测试准确率: 0.7832
✓ 保存最佳模型 (准确率: 0.7832)

...

训练完成！
最佳准确率: 0.8338
模型已保存到: ./sentiment_models/
```

### 3. 预测

**单条预测**:
```bash
python predict.py --model_path ./sentiment_models/best_model.pth \
                  --text "Amazing restaurant! The food was delicious."
```

输出：
```
文本: Amazing restaurant! The food was delicious.
情感: positive
置信度: 0.9823
概率分布: {'negative': 0.0045, 'neutral': 0.0132, 'positive': 0.9823}
```

**交互模式**:
```bash
python predict.py --model_path ./sentiment_models/best_model.pth --interactive
```

### 4. 代码集成

```python
from sentiment_analysis import SentimentPredictor

# 加载模型
predictor = SentimentPredictor('./sentiment_models/best_model.pth')

# 单条预测
result = predictor.predict("The food was absolutely delicious!")
print(f"情感: {result['sentiment']}")
print(f"置信度: {result['confidence']:.2%}")

# 批量预测
texts = [
    "Great food!",
    "Bad service",
    "It's okay"
]
results = predictor.predict_batch(texts)
for r in results:
    print(f"{r['text']} → {r['sentiment']}")
```

## 🏗️ 模型架构

```
输入文本
   ↓
Tokenizer (DistilBertTokenizerFast)
   ↓
DistilBERT Encoder (6层, 768维)
   ↓
[CLS] Token 表示
   ↓
Pre-classifier (Linear 768→768 + ReLU)
   ↓
Dropout (0.3)
   ↓
Classifier (Linear 768→3)
   ↓
Softmax → 概率分布 (negative, neutral, positive)
```

## 📊 性能

在餐厅评论数据集（9951条）上的表现：
- **测试准确率**: 83.38%
- **训练时间**: 10 epochs, ~15-20分钟 (GPU)
- **推理速度**: ~500 样本/秒

## 🔄 与公文核校系统的区别

| 特性 | 公文核校系统 | 情感分析模块 |
|------|------------|------------|
| **任务类型** | 序列标注 (Token Classification) | 文本分类 (Sequence Classification) |
| **模型** | BERT (中文) | DistilBERT (英文) |
| **标签** | BIO 标签 (O, B-ERROR, I-ERROR) | 3个类别 (negative, neutral, positive) |
| **输入** | 中文公文 | 英文评论 |
| **输出** | 每个 token 的标签 | 整个文本的情感类别 |
| **数据格式** | JSONL (token-level) | CSV (document-level) |

## 💡 高级功能

### 冻结基础层训练

适合小数据集或快速原型：

```bash
python train.py --csv_file data.csv --freeze_base --epochs 5
```

只训练分类头（~590K 参数），不更新 DistilBERT 基础层。

### 自定义模型参数

```python
from sentiment_analysis.model import CustomDistilBertForSequenceClassification

model = CustomDistilBertForSequenceClassification(
    num_labels=3,
    dropout_prob=0.3  # 调整 dropout 防止过拟合
)

# 查看参数统计
params = model.get_trainable_params()
print(f"总参数: {params['total']:,}")
print(f"可训练参数: {params['trainable']:,}")
```

## 📚 原始 Notebook 内容

从 Notebook 提取的主要内容：

1. **ReviewDataset 类** - 处理餐厅评论数据
2. **CustomDistilBertForSequenceClassification** - 自定义分类模型
3. **训练循环** - AdamW 优化器，CrossEntropyLoss
4. **评估** - 计算准确率
5. **预测函数** - `predict_sentiment()`

### 主要改进

相比原始 notebook：
- ✅ 模块化代码结构（独立的 .py 文件）
- ✅ 命令行参数支持
- ✅ 批量预测功能
- ✅ 交互式预测模式
- ✅ 详细的结果输出（置信度、概率分布）
- ✅ 模型保存/加载工具
- ✅ 完整的文档和示例
- ✅ 可作为 Python 包导入

## 🔧 依赖安装

```bash
cd sentiment_analysis
pip install -r requirements.txt
```

或手动安装：
```bash
pip install torch>=2.0.0 transformers>=4.30.0 pandas>=1.5.0 \
            scikit-learn>=1.3.0 tqdm>=4.65.0
```

## 🤔 何时使用这个模块？

- ✅ 需要对英文文本进行情感分析
- ✅ 评论、反馈、社交媒体文本分类
- ✅ 学习 DistilBERT 的实际应用
- ✅ 需要轻量级、快速的文本分类模型

## 📖 参考资源

- [DistilBERT 论文](https://arxiv.org/abs/1910.01108)
- [原始 Notebook](https://colab.research.google.com/drive/1B_ERSgQDLNOL8NPCvkn7s8h_vEdZ_szI)
- [HuggingFace DistilBERT 文档](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [餐厅评论数据集](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews)

---

**创建日期**: 2026-03-12  
**来源**: Google Colab notebook  
**适用场景**: 英文文本情感分析
