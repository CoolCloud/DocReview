"""
快速测试脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sentiment_analysis.predict import SentimentPredictor

print("加载模型...")
model_path = Path(__file__).parent / 'sentiment_models' / 'best_model.pth'

if not model_path.exists():
    print(f"❌ 模型不存在: {model_path}")
    print("请先训练模型或确认模型路径正确")
    sys.exit(1)

predictor = SentimentPredictor(str(model_path))

# 测试样本
test_samples = [
    "The food was absolutely delicious and the service was amazing!",
    "Terrible experience. The food was cold and service was awful.",
    "It was okay, nothing special really.",
    "Great ambience and friendly staff. Highly recommended!",
    "Worst restaurant ever. Will never come back."
]

print("\n" + "="*60)
print("情感分析推理测试")
print("="*60)

for i, text in enumerate(test_samples, 1):
    result = predictor.predict(text)
    print(f"\n{i}. 文本: {text}")
    print(f"   情感: {result['sentiment'].upper()}")
    print(f"   置信度: {result['confidence']:.2%}")
    print(f"   概率: neg={result['probabilities']['negative']:.2%}, "
          f"neu={result['probabilities']['neutral']:.2%}, "
          f"pos={result['probabilities']['positive']:.2%}")

print("\n" + "="*60)
print("测试完成！")
print("="*60)
