"""
MacBERT 公文校验 - 使用示例

演示如何使用 MacBERT 模块进行公文校验
"""
import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from doc_review_macbert import DocReviewPredictor


def example_single_prediction():
    """示例 1: 单文本预测"""
    print("\n" + "=" * 60)
    print("示例 1: 单文本预测")
    print("=" * 60 + "\n")
    
    # 加载模型
    predictor = DocReviewPredictor(
        model_path="models/best_model",
        device="auto"  # 自动选择最佳设备
    )
    
    # 测试文本
    texts = [
        "根据上级文件精神，认真组织学习。",  # 正确
        "根据上级文件精神，任真组织学习。",  # 错误：任真 -> 认真
        "为加强组织建设，坚定的推进工作。",  # 错误：坚定的 -> 坚定地
    ]
    
    for text in texts:
        result = predictor.predict(text, return_tokens=False)
        print(predictor.format_result(result))
        print()


def example_batch_prediction():
    """示例 2: 批量预测"""
    print("\n" + "=" * 60)
    print("示例 2: 批量预测")
    print("=" * 60 + "\n")
    
    predictor = DocReviewPredictor("models/best_model")
    
    # 批量文本
    texts = [
        "会议讨论并通过了工作方案，取的显著成效。",
        "各单位要高度重视，贯切落实相关要求。",
        "关与此事的通知已经发布，请各部门认真执行。",
        "为进一步提高工作质量，现将有关事项通知如下。",
    ]
    
    print(f"批量预测 {len(texts)} 条文本...\n")
    results = predictor.predict_batch(texts, batch_size=4)
    
    # 统计结果
    error_count = sum(1 for r in results if r['has_error'])
    print(f"结果统计:")
    print(f"  总数: {len(results)}")
    print(f"  含错误: {error_count}")
    print(f"  正确: {len(results) - error_count}")
    print()
    
    # 显示错误样本
    print("错误样本详情:")
    for i, result in enumerate(results):
        if result['has_error']:
            print(f"\n文本 {i+1}: {result['text']}")
            for err in result['errors']:
                print(f"  → 错误: 「{err['text']}」 位置: [{err['start']}:{err['end']}]")


def example_custom_analysis():
    """示例 3: 自定义分析"""
    print("\n" + "=" * 60)
    print("示例 3: 自定义错误分析")
    print("=" * 60 + "\n")
    
    predictor = DocReviewPredictor("models/best_model")
    
    text = "关与加强公文管理的通知已经下发，请各部门任真落实，坚定的推进相关工作。"
    
    result = predictor.predict(text, return_tokens=True)
    
    print(f"原文: {result['text']}")
    print(f"文本长度: {len(result['text'])} 字")
    print(f"错误数量: {len(result['errors'])}")
    print()
    
    if result['has_error']:
        print("错误详情:")
        for i, error in enumerate(result['errors'], 1):
            print(f"\n错误 {i}:")
            print(f"  片段: 「{error['text']}」")
            print(f"  位置: {error['start']} - {error['end']}")
            print(f"  Tokens: {' '.join(error['tokens'])}")
            
            # 提取上下文
            context_start = max(0, error['start'] - 5)
            context_end = min(len(text), error['end'] + 5)
            context = text[context_start:context_end]
            error_in_context = "..." + context + "..."
            print(f"  上下文: {error_in_context}")


def example_error_statistics():
    """示例 4: 错误统计分析"""
    print("\n" + "=" * 60)
    print("示例 4: 错误统计分析")
    print("=" * 60 + "\n")
    
    predictor = DocReviewPredictor("models/best_model")
    
    # 构造测试集
    test_cases = [
        {"text": "根据规定，任真完成任务。", "error_type": "错别字"},
        {"text": "坚定的推进工作。", "error_type": "的地得混用"},
        {"text": "取的显著成果。", "error_type": "同音字"},
        {"text": "贯切落实要求。", "error_type": "错别字"},
        {"text": "关与此事的通知。", "error_type": "同音字"},
        {"text": "认真完成任务。", "error_type": "正确"},
    ]
    
    # 批量预测
    texts = [case['text'] for case in test_cases]
    results = predictor.predict_batch(texts)
    
    # 统计
    error_types = {}
    for case, result in zip(test_cases, results):
        error_type = case['error_type']
        if error_type not in error_types:
            error_types[error_type] = {'total': 0, 'detected': 0}
        
        error_types[error_type]['total'] += 1
        if result['has_error']:
            error_types[error_type]['detected'] += 1
    
    # 显示统计
    print("检测统计:")
    print("-" * 60)
    for error_type, stats in error_types.items():
        total = stats['total']
        detected = stats['detected']
        rate = detected / total * 100 if total > 0 else 0
        print(f"{error_type:12s}: {detected}/{total} 检出 ({rate:.1f}%)")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("🤖 MacBERT 公文校验 - 使用示例")
    print("=" * 60)
    
    try:
        # 检查模型是否存在
        model_path = Path("models/best_model")
        if not model_path.exists():
            print("\n❌ 错误: 未找到训练好的模型")
            print(f"   请先运行: bash run_macbert.sh")
            print(f"   或手动训练: python doc_review_macbert/train.py")
            return
        
        # 运行示例
        example_single_prediction()
        input("\n按 Enter 继续下一个示例...")
        
        example_batch_prediction()
        input("\n按 Enter 继续下一个示例...")
        
        example_custom_analysis()
        input("\n按 Enter 继续下一个示例...")
        
        example_error_statistics()
        
        print("\n" + "=" * 60)
        print("✅ 所有示例运行完成！")
        print("=" * 60)
        print("\n📚 更多用法请参考: doc_review_macbert/README.md")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
