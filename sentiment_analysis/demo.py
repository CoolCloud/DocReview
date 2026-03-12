"""
情感分析 Demo
演示如何使用 DistilBERT 进行情感分析
"""
from predict import SentimentPredictor


def demo_basic_usage():
    """基础使用示例"""
    print("="*60)
    print("示例 1: 基础使用")
    print("="*60)
    
    # 假设已有训练好的模型
    model_path = "./sentiment_models/best_model.pth"
    
    try:
        # 创建预测器
        predictor = SentimentPredictor(model_path)
        
        # 测试评论
        reviews = [
            "The food was absolutely delicious! Great service too.",
            "Terrible experience. The food was cold and service was slow.",
            "It was okay, nothing special."
        ]
        
        for review in reviews:
            result = predictor.predict(review)
            print(f"\n评论: {result['text']}")
            print(f"情感: {result['sentiment']}")
            print(f"置信度: {result['confidence']:.2%}")
    
    except FileNotFoundError:
        print(f"⚠ 模型文件未找到: {model_path}")
        print("请先训练模型或下载预训练模型")


def demo_batch_prediction():
    """批量预测示例"""
    print("\n" + "="*60)
    print("示例 2: 批量预测")
    print("="*60)
    
    model_path = "./sentiment_models/best_model.pth"
    
    try:
        predictor = SentimentPredictor(model_path)
        
        reviews = [
            "Amazing restaurant! Best meal I've had in years.",
            "Not worth the price. Very disappointing.",
            "Average food, average service.",
            "Highly recommend! Will definitely come back.",
            "Worst dining experience ever."
        ]
        
        results = predictor.predict_batch(reviews)
        
        print(f"\n处理了 {len(results)} 条评论:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['sentiment'].upper()}] {result['text'][:50]}...")
    
    except FileNotFoundError:
        print(f"⚠ 模型文件未找到: {model_path}")


def demo_with_probability():
    """显示详细概率分布"""
    print("\n" + "="*60)
    print("示例 3: 详细概率分布")
    print("="*60)
    
    model_path = "./sentiment_models/best_model.pth"
    
    try:
        predictor = SentimentPredictor(model_path)
        
        text = "The ambience was good, food was quite good."
        result = predictor.predict(text)
        
        print(f"\n评论: {result['text']}")
        print(f"\n预测结果: {result['sentiment']} (置信度: {result['confidence']:.2%})")
        print(f"\n概率分布:")
        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {label:8s}: {prob:.2%} {bar}")
    
    except FileNotFoundError:
        print(f"⚠ 模型文件未找到: {model_path}")


def demo_training_workflow():
    """训练工作流示例（代码示例）"""
    print("\n" + "="*60)
    print("示例 4: 训练工作流")
    print("="*60)
    
    print("""
训练步骤:

1. 准备数据（CSV 格式，包含 'Review' 和 'Rating' 列）:
   
   Review,Rating
   "Great food!",positive
   "Bad service",negative
   "It's okay",neutral

2. 运行训练脚本:
   
   python train.py --csv_file data.csv --epochs 10 --batch_size 16

3. 使用训练好的模型:
   
   from predict import SentimentPredictor
   predictor = SentimentPredictor('./sentiment_models/best_model.pth')
   result = predictor.predict("Amazing restaurant!")
   print(result['sentiment'])  # 'positive'

4. (可选) 下载示例数据:
   
   wget https://raw.githubusercontent.com/kyuz0/llm-chronicles/main/datasets/restaurant_reviews.csv
    """)


def main():
    """运行所有示例"""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "DistilBERT 情感分析 Demo" + " "*19 + "║")
    print("╚" + "═"*58 + "╝")
    
    demo_basic_usage()
    demo_batch_prediction()
    demo_with_probability()
    demo_training_workflow()
    
    print("\n" + "="*60)
    print("Demo 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
