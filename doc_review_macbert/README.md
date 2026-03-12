# MacBERT 公文校验模块

基于 MacBERT 的中文公文错误检测系统，专门用于检测公文中的常见错误。

## 📋 项目简介

MacBERT (Mac as correction BERT) 是哈工大讯飞联合实验室推出的中文预训练模型，针对中文任务进行了优化。本模块使用 MacBERT 进行序列标注任务，检测公文中的各类错误。

### ✨ 特性

- ✅ **基于 MacBERT** - 使用针对中文优化的预训练模型
- ✅ **序列标注** - BIO 标注方式精确定位错误
- ✅ **多种错误类型** - 检测错别字、的地得混用、同音字错误等
- ✅ **GPU 加速** - 支持 Mac GPU (MPS)、CUDA 和 CPU
- ✅ **易于使用** - 提供命令行工具和 Python API
- ✅ **完整流程** - 数据生成、训练、预测一站式解决

### 🎯 支持的错误类型

1. **错别字** - 形近字混淆（认真/任真、贯彻/贯切）
2. **的地得混用** - 的地得使用错误
3. **同音字错误** - 同音字混淆（取得/取的、关于/关与）
4. **标点符号** - 标点使用不规范
5. **非正式用语** - 口语化表达
6. **词语搭配** - 搭配不当（提高/增加）

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 或手动安装
pip install torch transformers scikit-learn tqdm numpy
```

### 2. 生成训练数据

```bash
# 生成默认数据集（800 训练 + 200 测试）
python doc_review_macbert/generate_data.py

# 自定义参数
python doc_review_macbert/generate_data.py \
    --train 1000 \
    --test 200 \
    --error-rate 0.4 \
    --output-dir doc_review_macbert/data
```

生成的数据格式（JSONL）：
```json
{"id": "train_1", "text": "根据上级文件精神，任真组织学习。", "labels": ["O", "O", "O", ...], "has_error": true}
```

### 3. 训练模型

```bash
# 使用默认参数训练
python doc_review_macbert/train.py

# 自定义参数
python doc_review_macbert/train.py \
    --model-name hfl/chinese-macbert-base \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --output-dir doc_review_macbert/models
```

**训练参数说明：**
- `--model-name`: 预训练模型
  - `hfl/chinese-macbert-base` (推荐，102M 参数)
  - `hfl/chinese-macbert-large` (324M 参数)
- `--epochs`: 训练轮数（默认 5）
- `--batch-size`: 批次大小（默认 16，GPU 内存不足可减小）
- `--learning-rate`: 学习率（默认 2e-5）

### 4. 预测

#### 演示模式（推荐新手）

```bash
python doc_review_macbert/predict.py --mode demo
```

#### 交互模式

```bash
python doc_review_macbert/predict.py --mode interactive
```

#### 批量预测

```bash
# 从文件批量预测
python doc_review_macbert/predict.py \
    --mode batch \
    --input-file test_texts.txt \
    --output-file results.json
```

### 5. Python API 使用

```python
from doc_review_macbert import DocReviewPredictor

# 加载模型
predictor = DocReviewPredictor(
    model_path="doc_review_macbert/models/best_model",
    device="auto"  # 自动选择设备
)

# 单文本预测
text = "根据上级文件精神，任真组织学习。"
result = predictor.predict(text)

print(f"原文: {result['text']}")
print(f"有错误: {result['has_error']}")
if result['has_error']:
    for error in result['errors']:
        print(f"  错误: {error['text']} (位置: {error['start']}-{error['end']})")

# 批量预测
texts = [
    "根据上级文件精神，认真组织学习。",
    "为加强组织建设，坚定的推进工作。",
    "会议讨论并通过了工作方案，取的显著成效。"
]
results = predictor.predict_batch(texts)
```

## 📁 项目结构

```
doc_review_macbert/
├── __init__.py           # 模块初始化
├── model.py              # MacBERT 模型定义
├── dataset.py            # 数据集加载和处理
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── generate_data.py      # 数据生成脚本
├── requirements.txt      # 依赖列表
└── README.md             # 本文档
```

## 🔧 高级用法

### 自定义错误模式

编辑 `generate_data.py` 中的 `ERROR_PATTERNS` 字典：

```python
ERROR_PATTERNS = {
    "your_error_type": {
        "description": "错误描述",
        "examples": [
            {"correct": "正确写法", "wrong": "错误写法"},
            # 添加更多示例...
        ]
    }
}
```

### 模型微调

如需在特定领域数据上微调：

```bash
# 使用已训练的模型继续训练
python doc_review_macbert/train.py \
    --model-name doc_review_macbert/models/best_model \
    --train-data your_domain_data/train.jsonl \
    --test-data your_domain_data/test.jsonl \
    --epochs 3
```

### 模型评估

```python
from doc_review_macbert.train import evaluate
from doc_review_macbert.dataset import create_data_loaders
from transformers import BertTokenizer

# 加载模型和数据
model = MacBERTForDocReview.from_pretrained("doc_review_macbert/models/best_model")
tokenizer = BertTokenizer.from_pretrained("doc_review_macbert/models/best_model")
_, test_loader = create_data_loaders(
    "doc_review_macbert/data/train.jsonl",
    "doc_review_macbert/data/test.jsonl",
    tokenizer
)

# 评估
device = torch.device("mps")  # 或 "cuda", "cpu"
model.to(device)
loss, acc, f1, precision, recall, report = evaluate(model, test_loader, device)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(report)
```

## 📊 性能指标

在默认生成的数据集上（800 训练 + 200 测试，40% 错误率）：

| 指标 | 预期值 |
|------|--------|
| 准确率 | > 98% |
| F1 分数 | > 95% |
| 精确率 | > 95% |
| 召回率 | > 95% |

*实际性能取决于数据质量和训练参数*

## 💡 使用建议

1. **数据质量至关重要**
   - 确保训练数据覆盖真实场景中的错误类型
   - 错误样本比例建议 30%-50%
   - 定期使用真实数据评估和迭代

2. **模型选择**
   - 一般场景：`hfl/chinese-macbert-base` 足够
   - 高精度需求：使用 `hfl/chinese-macbert-large`
   - 计算资源受限：减小 batch_size 或使用 base 版本

3. **训练技巧**
   - 学习率：2e-5 到 5e-5 之间效果较好
   - Epoch：3-5 轮通常足够，避免过拟合
   - 使用验证集监控训练过程

4. **部署建议**
   - 生产环境建议使用 GPU 加速
   - 可使用 ONNX 或 TorchScript 优化推理速度
   - 批量预测时设置合适的 batch_size

## 🔍 常见问题

**Q: 为什么选择 MacBERT 而不是 BERT？**

A: MacBERT 针对中文进行了特殊优化，在中文 NLP 任务上通常表现更好。它使用同义词替换代替 [MASK]，更适合中文文本理解。

**Q: 如何处理超长文本？**

A: 模型默认支持 256 个字符。超长文本可以：
1. 增加 `max_length` 参数（会增加内存消耗）
2. 分段处理后合并结果
3. 使用滑动窗口方法

**Q: 模型能检测所有类型的错误吗？**

A: 模型只能检测训练数据中出现过的错误模式。建议根据实际需求定制训练数据。

**Q: Mac GPU (MPS) 训练速度如何？**

A: 在 M1/M2/M3 芯片上，MPS 可以提供 3-5 倍于 CPU 的训练速度，但不如专业 GPU（如 RTX 系列）快。

## 📚 参考资料

- [MacBERT 论文](https://arxiv.org/abs/2004.13922)
- [MacBERT GitHub](https://github.com/ymcui/MacBERT)
- [Hugging Face 文档](https://huggingface.co/docs/transformers)
- [序列标注教程](https://huggingface.co/docs/transformers/tasks/token_classification)

## 📧 反馈与贡献

如有问题或建议，欢迎提出 Issue 或 Pull Request。

## 📄 许可证

MIT License

---

**注意**: 本模块生成的训练数据为模拟数据，实际使用时建议使用真实的公文数据进行训练以获得更好的效果。
