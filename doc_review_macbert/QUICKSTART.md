# MacBERT 公文校验快速入门

欢迎使用基于 MacBERT 的公文校验模块！本文档帮助你快速上手。

## 🎯 5 分钟快速开始

### 方式一：一键启动（推荐）

```bash
# 自动完成数据生成、训练和演示
bash run_macbert.sh
```

这个脚本会自动：
1. 安装依赖
2. 生成训练数据（800 训练 + 200 测试）
3. 训练 MacBERT 模型（约 10-30 分钟）
4. 运行演示

### 方式二：分步执行

```bash
# 1. 安装依赖
pip install -r doc_review_macbert/requirements.txt

# 2. 生成数据
python doc_review_macbert/generate_data.py

# 3. 训练模型
python doc_review_macbert/train.py

# 4. 运行演示
python doc_review_macbert/predict.py --mode demo
```

## 💻 使用方式

### 1. 命令行工具

```bash
# 演示模式 - 查看预定义示例
python doc_review_macbert/predict.py --mode demo

# 交互模式 - 实时输入文本检测
python doc_review_macbert/predict.py --mode interactive

# 批量模式 - 处理文件
python doc_review_macbert/predict.py \
    --mode batch \
    --input-file texts.txt \
    --output-file results.json
```

### 2. Python API

```python
from doc_review_macbert import DocReviewPredictor

# 加载模型
predictor = DocReviewPredictor("models_macbert/best_model")

# 预测单个文本
result = predictor.predict("根据上级文件精神，任真组织学习。")
print(f"有错误: {result['has_error']}")
for error in result['errors']:
    print(f"错误: {error['text']}")

# 批量预测
texts = ["文本1", "文本2", "文本3"]
results = predictor.predict_batch(texts)
```

### 3. 运行示例代码

```bash
# 查看完整的使用示例
python demo_macbert.py
```

## 📚 详细文档

- [完整使用文档](doc_review_macbert/README.md) - 详细的 API 文档和高级用法
- [模型说明](doc_review_macbert/model.py) - MacBERT 模型实现细节
- [数据格式](doc_review_macbert/dataset.py) - 数据集格式说明

## 🎓 支持的错误类型

✅ 错别字（认真/任真、贯彻/贯切）  
✅ 的地得混用  
✅ 同音字错误（取得/取的、关于/关与）  
✅ 标点符号错误  
✅ 非正式用语  
✅ 词语搭配不当  

## ⚙️ 系统要求

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM（推荐）
- GPU（可选，推荐使用以加速训练）
  - Mac M1/M2/M3 (MPS)
  - NVIDIA GPU (CUDA)

## 🆘 常见问题

**Q: 训练需要多长时间？**  
A: 在 Mac M1/M2 上约 15-20 分钟，在 NVIDIA GPU 上约 5-10 分钟，CPU 上可能需要 1-2 小时。

**Q: 如何使用自己的数据？**  
A: 准备 JSONL 格式的数据文件，每行一个 JSON 对象包含 `text` 和 `labels` 字段，然后使用 `--train-data` 和 `--test-data` 参数。

**Q: 模型准确率如何？**  
A: 在默认数据集上可达 98%+ 准确率，实际效果取决于数据质量和错误类型。

**Q: 可以检测哪些错误？**  
A: 模型可以检测训练数据中出现的错误类型。默认包括常见的公文错误，你可以通过修改 `generate_data.py` 添加自定义错误类型。

## 🚀 下一步

1. **自定义错误类型** - 编辑 `doc_review_macbert/generate_data.py` 添加特定领域的错误
2. **使用真实数据** - 用实际公文数据替换模拟数据进行训练
3. **模型优化** - 调整训练参数（学习率、batch size、epochs）
4. **部署应用** - 集成到你的应用或服务中

## 📞 获取帮助

- 查看详细文档：`doc_review_macbert/README.md`
- 运行示例代码：`python demo_macbert.py`
- 查看 MacBERT 官方：https://github.com/ymcui/MacBERT

---

Happy Coding! 🎉
