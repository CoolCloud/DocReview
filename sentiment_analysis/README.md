# DistilBERT 情感分析模块

从 Google Colab notebook 提取并改进的独立情感分析模块。

## 📋 项目简介

这是一个基于 DistilBERT 的情感分析系统，用于对文本（如餐厅评论）进行三分类：
- **Negative** (负面)
- **Neutral** (中性)
- **Positive** (正面)

### 特点

- ✅ 完整的训练、评估和预测流程
- ✅ 自定义 DistilBERT 分类模型
- ✅ 支持 CPU、CUDA GPU、Mac GPU (MPS)
- ✅ 批量预测支持
- ✅ 交互式预测模式
- ✅ 可选的基础层冻结（适合小数据集）

## 📁 项目结构

```
sentiment_analysis/
├── dataset.py      # 数据集类和数据加载
├── model.py        # 自定义 DistilBERT 模型
├── train.py        # 训练脚本
├── predict.py      # 预测脚本
├── demo.py         # 使用示例
└── README.md       # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch transformers pandas scikit-learn tqdm
```

### 2. 准备数据

数据格式：CSV 文件，包含 `Review` 和 `Rating` 两列

```csv
Review,Rating
"The food was amazing!",positive
"Terrible service",negative
"It was okay",neutral
```

**下载示例数据集**:
```bash
wget https://raw.githubusercontent.com/kyuz0/llm-chronicles/main/datasets/restaurant_reviews.csv
```

### 3. 训练模型

```bash
cd sentiment_analysis
python train.py --csv_file restaurant_reviews.csv --epochs 10 --batch_size 16
```

**训练参数**:
- `--csv_file`: 训练数据路径（必需）
- `--output_dir`: 模型保存目录（默认: `./sentiment_models`）
- `--epochs`: 训练轮数（默认: 10）
- `--batch_size`: Batch 大小（默认: 16）
- `--lr`: 学习率（默认: 5e-5）
- `--max_length`: 最大序列长度（默认: 512）
- `--train_split`: 训练集比例（默认: 0.8）
- `--freeze_base`: 是否冻结 DistilBERT 基础层

### 4. 预测

**单条预测**:
```bash
python predict.py --model_path ./sentiment_models/best_model.pth \
                  --text "The food was delicious and service was great!"
```

**交互模式**:
```bash
python predict.py --model_path ./sentiment_models/best_model.pth --interactive
```

### 5. 代码示例

```python
from predict import SentimentPredictor

# 加载模型
predictor = SentimentPredictor('./sentiment_models/best_model.pth')

# 单条预测
result = predictor.predict("Amazing restaurant!")
print(result['sentiment'])      # 'positive'
print(result['confidence'])     # 0.9823

# 批量预测
texts = [
    "Great food!",
    "Bad service",
    "It's okay"
]
results = predictor.predict_batch(texts)
for r in results:
    print(f"{r['text']} → {r['sentiment']} ({r['confidence']:.2%})")
```

## 🏗️ 模型架构

```
DistilBERT (预训练)
    ↓
[CLS] Token 表示
    ↓
Pre-classifier (Linear 768→768 + ReLU)
    ↓
Dropout (0.3)
    ↓
Classifier (Linear 768→3)
    ↓
Logits (negative, neutral, positive)
```

### 参数统计
- **总参数**: ~66M
- **DistilBERT**: ~66M
- **分类头**: ~590K

## 📊 性能

在餐厅评论数据集上的结果（Colab notebook）:
- **测试准确率**: 83.38%
- **训练时长**: ~10 epochs
- **设备**: Google Colab GPU

## 🔧 高级功能

### 冻结基础层训练

适用于**数据量较小**或需要**快速训练**的场景：

```bash
python train.py --csv_file data.csv --freeze_base --epochs 5
```

只训练分类头，大幅减少训练时间。

### 自定义模型

```python
from model import CustomDistilBertForSequenceClassification

# 创建模型
model = CustomDistilBertForSequenceClassification(
    num_labels=3,
    dropout_prob=0.3
)

# 冻结基础层
model.freeze_base_model()

# 查看参数统计
params = model.get_trainable_params()
print(f"可训练参数: {params['trainable']:,}")
```

### 使用 HuggingFace Transformers

如果想使用 HuggingFace 的 `Trainer` API:

```python
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
```

## 📚 数据格式

### CSV 格式

```csv
Review,Rating
"The ambience was good food was quite good.",positive
"Terrible experience. Will never come back.",negative
"It's fine, nothing special.",neutral
```

### 标签映射

```python
label_dict = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}
```

支持数字或文本标签，会自动转换。

## 🛠️ 故障排除

### 1. 内存不足

减小 batch size 或 max_length:
```bash
python train.py --csv_file data.csv --batch_size 8 --max_length 256
```

### 2. 训练过慢

使用冻结基础层训练:
```bash
python train.py --csv_file data.csv --freeze_base
```

### 3. 模型未找到

确保模型路径正确:
```python
import os
model_path = "./sentiment_models/best_model.pth"
print(f"模型存在: {os.path.exists(model_path)}")
```

## 📖 原始 Notebook

本模块基于以下 Colab notebook:
- **标题**: DistilBERT For Sentence Classification
- **链接**: https://colab.research.google.com/drive/1B_ERSgQDLNOL8NPCvkn7s8h_vEdZ_szI
- **任务**: 餐厅评论情感分析
- **数据集**: Kaggle Restaurant Reviews

### 主要改进

相比原始 notebook，本模块增加了:
- ✅ 完整的模块化代码结构
- ✅ 命令行参数支持
- ✅ 批量预测功能
- ✅ 交互式预测模式
- ✅ 更详细的结果输出（置信度、概率分布）
- ✅ 模型保存和加载工具函数
- ✅ 完整的文档和示例

## 🔗 相关资源

- [DistilBERT 论文](https://arxiv.org/abs/1910.01108)
- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [原始数据集](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews)

## 📝 License

本项目基于 Colab notebook 改进，用于学习和研究目的。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**: 这是一个从 Colab notebook 提取的独立模块，与项目中的公文核校系统（基于 BERT 的序列标注）是两个不同的任务。
