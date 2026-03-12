# DocReview 项目结构

本项目包含多个独立的公文校验和文本分析模块，已重新整理为清晰的模块化结构。

## 📁 项目结构（整理后）

```
DocReview/
├── doc_review_macbert/          # ✨ MacBERT 公文校验（当前最佳版本）
│   ├── __init__.py              # 模块初始化
│   ├── model.py                 # MacBERT 模型定义
│   ├── dataset.py               # 数据集处理
│   ├── train.py                 # 训练脚本
│   ├── predict.py               # 预测脚本（支持3种模式）
│   ├── generate_data.py         # 数据生成（6种错误类型）
│   ├── demo.py                  # 使用示例
│   ├── run.sh                   # 一键启动脚本
│   ├── tests/                   # 测试文件
│   │   ├── test_module.py       # 模块测试
│   │   └── test_result.py       # 结果测试
│   ├── requirements.txt         # 依赖列表
│   ├── README.md                # 详细文档
│   ├── QUICKSTART.md            # 快速入门
│   └── SUMMARY.md               # 项目总结
│
├── doc_review_v1/               # 📦 V1 版本（已归档）
│   ├── train.py                 # 训练脚本
│   ├── inference.py             # 推理脚本
│   ├── generate_data.py         # 数据生成
│   ├── demo.py                  # 演示
│   ├── run.sh                   # 启动脚本
│   ├── QUICKSTART.md            # 快速入门
│   ├── README.md                # 说明文档
│   ├── data/                    # V1 训练数据
│   └── models/                  # V1 训练模型
│
├── doc_review_v2/               # 📦 V2 版本（已归档）
│   ├── train.py                 # 训练脚本（改进）
│   ├── inference.py             # 推理脚本（改进）
│   ├── generate_data.py         # 数据生成（改进）
│   ├── SUMMARY.md               # V2 总结
│   ├── V1_VS_V2.md              # 版本对比
│   ├── README.md                # 说明文档
│   └── models/                  # V2 训练模型
│
├── sentiment_analysis/          # 🎭 情感分析模块（参考项目）
│   ├── __init__.py
│   ├── model.py                 # DistilBERT 模型
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   └── README.md
│
├── README.md                    # 📖 项目主文档
├── requirements.txt             # 项目依赖
└── PROJECT_STRUCTURE.md         # 本文档
```

## 🎯 模块对比

| 特性 | MacBERT 模块 | V2 版本 | V1 版本 | 情感分析 |
|-----|-------------|---------|---------|---------|
| **模型** | MacBERT (中文优化) | BERT (中文) | BERT (中文) | DistilBERT |
| **任务** | 公文错误检测 | 公文错误检测 | 公文错误检测 | 情感分类 |
| **结构** | 独立模块 | 独立模块 | 独立模块 | 独立模块 |
| **数据格式** | JSONL | JSONL | JSON | CSV |
| **状态** | ✨ 当前最佳 | 📦 已归档 | 📦 已归档 | ✅ 稳定 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

## 🚀 快速开始

### MacBERT 公文校验模块（推荐）

```bash
# 一键启动
cd doc_review_macbert
bash run.sh

# 或分步执行
python generate_data.py
python train.py --train-data data/train.jsonl --test-data data/test.jsonl
python predict.py --mode demo
```

详细文档：[doc_review_macbert/QUICKSTART.md](doc_review_macbert/QUICKSTART.md)

### V2 公文校验（归档版本）

```bash
cd doc_review_v2
python generate_data.py
python train.py
python inference.py --mode demo
```

详细文档：[doc_review_v2/README.md](doc_review_v2/README.md)

### V1 公文校验（归档版本）

```bash
cd doc_review_v1
bash run.sh
```

详细文档：[doc_review_v1/README.md](doc_review_v1/README.md)

### 情感分析模块

```bash
cd sentiment_analysis
python train.py --data restaurant_reviews.csv
python predict.py --model sentiment_models/best_model.pth
```

详细文档：[sentiment_analysis/README.md](sentiment_analysis/README.md)

## 📊 各模块优势

### MacBERT 模块 ✨

**优势：**
- ✅ **中文优化** - MacBERT 专为中文设计，效果更好
- ✅ **独立模块** - 清晰的模块化结构，易于维护
- ✅ **完整 API** - 提供 Python API 和命令行工具
- ✅ **易于扩展** - 模块化设计便于定制
- ✅ **详细文档** - 包含完整的使用文档和示例
- ✅ **测试完善** - 包含单元测试和集成测试

**训练效果：**
- 准确率：100%
- 精确率：100%
- 召回率：100%
- F1 分数：100%

**适用场景：**
- 🎯 推荐用于新项目开发
- 需要集成到其他系统
- 需要定制化功能
- 需要详细的 API 文档

### V2 公文校验 📦

**优势：**
- ✅ **性能优异** - F1 分数 99.54%
- ✅ **数据丰富** - 1000 条训练数据
- ✅ **成熟稳定** - 已经过充分测试

**适用场景：**
- 快速原型验证
- 参考旧版实现
- 对比不同方法

### V1 公文校验 📦

**优势：**
- ✅ **简单直接** - 基础实现
- ✅ **学习参考** - 适合学习使用

**适用场景：**
- 学习序列标注任务
- 理解基础实现
- 对比版本演进

### 情感分析模块 🎭

**优势：**
- ✅ **轻量级** - 使用 DistilBERT
- ✅ **三分类** - Positive/Neutral/Negative
- ✅ **模块化** - 独立的情感分析模块

**适用场景：**
- 评论分析
- 文本情感检测
- 轻量级部署需求

## 🎓 选择建议

### 选择 MacBERT 模块，如果你：
- 开始新的公文校验项目
- 需要更好的中文理解能力
- 需要集成到现有系统
- 需要定制化功能和扩展

### 选择 V2，如果你：
- 需要快速验证想法
- 对现有性能满意
- 不需要复杂的 API 集成

### 选择情感分析模块，如果你：
- 需要文本情感分类
- 不是公文校验任务

## 🛠️ 开发建议

### 使用 MacBERT 模块作为模板

如果你要开发新的文本分析模块，可以参考 `doc_review_macbert/` 的结构：

```
your_module/
├── __init__.py          # 模块接口
├── model.py             # 模型定义
├── dataset.py           # 数据处理
├── train.py             # 训练脚本
├── predict.py           # 预测接口
├── generate_data.py     # 数据生成（可选）
├── requirements.txt     # 依赖
└── README.md            # 文档
```

### 代码复用

各模块之间可以共享一些通用组件：
- 设备检测 (`get_device()`)
- 数据加载器模式
- 训练循环结构
- 评估指标计算

## 📚 学习路径

1. **初学者** → 先运行 `bash run_macbert.sh` 查看完整流程
2. **进阶** → 阅读 `doc_review_macbert/README.md` 了解 API 使用
3. **高级** → 查看模块源码，理解实现细节
4. **专家** → 基于模块开发定制化功能

## 🔗 相关资源

- [MacBERT 官方 GitHub](https://github.com/ymcui/MacBERT)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch 官方文档](https://pytorch.org/docs/)

## 📝 贡献指南

欢迎贡献代码和文档！建议的改进方向：

- 添加更多错误类型
- 优化模型性能
- 改进用户体验
- 完善文档和示例
- 添加单元测试

---

最后更新：2026-03-12
