# 公文校验 V2 版本

这是改进后的公文校验实现，相比 V1 版本有多项优化。

## 文件说明

- `train.py` - 训练脚本（改进版）
- `generate_data.py` - 数据生成脚本（改进版）
- `inference.py` - 推理脚本（改进版）
- `SUMMARY.md` - V2 版本总结
- `V1_VS_V2.md` - V1 与 V2 对比说明

## 目录结构

- `models/` - 训练好的模型

## V2 改进内容

详见 [SUMMARY.md](SUMMARY.md) 和 [V1_VS_V2.md](V1_VS_V2.md)

## 使用方法

```bash
# 生成数据
python generate_data.py --output-dir data --train-size 800 --test-size 200

# 训练模型
python train.py --train-data data/train.jsonl --test-data data/test.jsonl

# 推理
python inference.py --mode demo --model-path models/best_model
```

## 下一代版本

更强大的 MacBERT 版本详见 [doc_review_macbert](../doc_review_macbert/)
