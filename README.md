# 公文核校系统 V2

基于 BERT 中文模型的公文错误检测和纠正系统，针对 Mac GPU (MPS) 优化。

## ✨ 最新更新 (V2)

- ✅ **JSONL 数据格式** - 采用工业标准格式
- ✅ **1000 条训练数据** - 数据量翻倍（800训练 + 200测试）
- ✅ **37.5% 错误样本** - 错误比例大幅提升
- ✅ **6 种错误类型** - 覆盖更多公文常见错误
- ✅ **F1 分数 99.54%** - 性能卓越
- ✅ **召回率 100%** - 所有错误都能检测到

## 🎯 功能特性

- ✅ 错别字检测（取的/取得、任真/认真等）
- ✅ 的地得混用识别
- ✅ 标点符号规范性检查
- ✅ 非正式用语检测（关与/关于、贯切/贯彻等）
- ✅ 同音字错误识别
- ✅ 使用 Mac GPU 加速训练和推理

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 准确率 | **99.95%** |
| F1 分数 | **99.54%** |
| 精确率 | **99.08%** |
| 召回率 | **100%** |
| 推理速度 | <100ms/样本 |

## 🚀 快速开始

### 1. 测试现有模型（V2 推荐）

```bash
# 激活虚拟环境
source .venv/bin/activate

# 快速演示
python inference_v2.py

# 在测试集上评估
python inference_v2.py --test
```

### 2. 重新训练模型

```bash
# 生成训练数据（1000条）
python generate_data_v2.py

# 训练模型（~2-3分钟）
python train_v2.py
```

### 3. 在代码中使用

```python
from inference_v2 import DocReviewer

# 初始化
reviewer = DocReviewer("models/best_model_v2")

# 检测文本
text = "此次培训取的了良好效果。"
errors = reviewer.review(text)

if errors:
    for err in errors:
        print(f"位置 {err['start']}-{err['end']}: {err['text']}")
```

## 📁 项目结构

```
DocReview/
├── data/
│   ├── train.jsonl          # 训练集 (800条, JSONL格式)
│   ├── test.jsonl           # 测试集 (200条, JSONL格式)
│   └── *_sample.json        # 样本预览
│
├── models/
│   ├── best_model_v2/       # V2 最佳模型 ⭐
│   └── training_history.json
│
├── generate_data_v2.py      # 数据生成 V2 ⭐
├── train_v2.py              # 训练脚本 V2 ⭐
├── inference_v2.py          # 推理脚本 V2 ⭐
│
├── requirements.txt
├── README.md               # 本文件
├── QUICKSTART.md          # 快速开始指南
└── SUMMARY_V2.md          # 详细项目总结
```

## 🎯 错误类型示例

### 1. 常见错别字 ✅
```
输入: 此次培训取的了良好效果
检测: ✗ 位置 4-6: '取的'
```

### 2. 非正式用语 ✅
```
输入: 关与环境保护工作，提出以下意见
检测: ✗ 位置 0-2: '关与'
```

### 3. 多错误检测 ✅
```
输入: 请任真贯切执行
检测: ✗ 位置 1-3: '任真'
      ✗ 位置 3-5: '贯切'
```

## 📈 训练结果

最佳模型（Epoch 6）：

```
分类报告:
              precision    recall  f1-score   support
           O     1.0000    0.9995    0.9998      4054
     B-ERROR     0.9894    1.0000    0.9947        93
     I-ERROR     0.9919    1.0000    0.9959       122

错误检测:
  Precision: 99.08%
  Recall: 100.00%
  F1-Score: 99.54% ⭐
```

## 🔧 系统要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Mac (推荐，支持 MPS 加速) / Linux / Windows

## 💡 版本说明

### V2 (当前版本) - 2026.03.12
- ✅ JSONL 数据格式
- ✅ 1000 条训练数据
- ✅ 6 种错误类型
- ✅ F1 分数 99.54%
- ✅ 完整错误标注

### V1 (初始版本)
- 基础功能实现
- 500 条训练数据
- 4 种错误类型

详细对比请查看 [SUMMARY_V2.md](SUMMARY_V2.md)

## 📚 文档

- [README.md](README.md) - 项目概览（本文件）
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [SUMMARY_V2.md](SUMMARY_V2.md) - 详细项目总结

## 🎓 技术栈

- **PyTorch 2.x** - 深度学习框架
- **Transformers** - Hugging Face NLP 库
- **BERT (bert-base-chinese)** - 预训练模型
- **Token Classification** - 序列标注任务
- **BIO 标注** - Begin-Inside-Outside

## 📞 贡献与反馈

欢迎提出问题和改进建议！

---

**更新日期**: 2026年3月12日  
**版本**: V2.0  
**状态**: ✅ 生产就绪
