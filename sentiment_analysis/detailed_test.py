#!/usr/bin/env python3
"""详细推理测试"""
from predict import SentimentPredictor

print("="*60)
print("情感分析推理测试")
print("="*60)

# 加载模型
print("\n加载模型...")
predictor = SentimentPredictor('./sentiment_models/best_model.pth')

# 测试样本
test_samples = [
    ('Amazing food and great service!', 'positive'),
    ('Worst restaurant experience ever.', 'negative'),
    ('Food was average, nothing special.', 'neutral'),
    ('Highly recommend this place!', 'positive'),
    ('Bad food, slow service.', 'negative'),
    ('It was fine, not great not bad.', 'neutral'),
    ('The ambience was good food was quite good.', 'positive'),
    ('Terrible experience, will never come back.', 'negative'),
]

print(f"\n测试 {len(test_samples)} 个样本...\n")

correct = 0
for i, (text, expected) in enumerate(test_samples, 1):
    result = predictor.predict(text)
    pred = result['sentiment']
    conf = result['confidence']
    match = '✓' if pred == expected else '✗'
    correct += (pred == expected)
    
    print(f"{i}. {match} [{pred.upper()}] (期望: {expected}, 置信度: {conf:.1%})")
    print(f"   {text[:60]}...")
    
print("\n" + "="*60)
print(f"准确率: {correct}/{len(test_samples)} ({correct/len(test_samples)*100:.1f}%)")
print("="*60)

# 批量预测测试
print("\n批量预测测试...")
batch_texts = [
    "Excellent food!",
    "Very disappointed.",
    "It's okay I guess.",
]

results = predictor.predict_batch(batch_texts)
print(f"批量处理 {len(results)} 条评论:")
for r in results:
    print(f"  {r['sentiment']:8s} ({r['confidence']:.0%}) - {r['text']}")

print("\n测试完成！")
