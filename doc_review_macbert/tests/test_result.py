#!/usr/bin/env python3
"""批量测试 MacBERT 模型效果"""

from doc_review_macbert.predict import DocReviewPredictor

def main():
    # 加载模型
    predictor = DocReviewPredictor(model_path="../models/best_model")
    
    # 测试样例
    test_cases = [
        {
            "text": "根据上级文件精神，认真组织学习。",
            "description": "正确样例",
            "has_error": False
        },
        {
            "text": "根据上级文件精神，任真组织学习。",
            "description": "错别字: 任真 -> 认真", 
            "has_error": True,
            "error_text": "任真"
        },
        {
            "text": "为加强组织建设，经研究决定，坚定的推进工作。",
            "description": "的地得混用: 坚定的 -> 坚定地",
            "has_error": True,
            "error_text": "坚定的"
        },
        {
            "text": "会议讨论并通过了工作方案，取的了显著成效。",
            "description": "同音字错误: 取的 -> 取得",
            "has_error": True,
            "error_text": "取的"
        },
        {
            "text": "各单位要高度重视，贯切落实相关要求。",
            "description": "错别字: 贯切 -> 贯彻",
            "has_error": True,
            "error_text": "贯切"
        },
        {
            "text": "关与此事的通知已经发布，请各部门认真执行。",
            "description": "同音字错误: 关与 -> 关于",
            "has_error": True,
            "error_text": "关与"
        }
    ]
    
    print("\n" + "=" * 80)
    print("🎯 MacBERT 公文校验 - 批量测试结果")
    print("=" * 80)
    
    correct = 0
    total_errors = 0
    detected_errors = 0
    false_positives = 0
    
    for i, case in enumerate(test_cases, 1):
        result = predictor.predict(case['text'], return_tokens=False)
        has_detected_error = len(result['errors']) > 0
        
        # 判断是否正确
        is_correct = has_detected_error == case['has_error']
        
        print(f"\n{'='*80}")
        print(f"样例 {i}: {case['description']}")
        print(f"文本: {case['text']}")
        print(f"预期: {'有错误' if case['has_error'] else '无错误'}")
        print(f"检测: {'有错误' if has_detected_error else '无错误'}")
        
        if has_detected_error:
            print(f"检测到的错误: {', '.join([e['text'] for e in result['errors']])}")
        
        if is_correct:
            print("✅ 判断正确")
            correct += 1
        else:
            print("❌ 判断错误")
        
        # 统计
        if case['has_error']:
            total_errors += 1
            if has_detected_error:
                detected_errors += 1
        elif has_detected_error:
            false_positives += 1
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    print(f"总样例数: {len(test_cases)}")
    print(f"判断正确: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
    print(f"\n错误检测:")
    print(f"  - 应检测错误数: {total_errors}")
    print(f"  - 成功检测: {detected_errors}")
    print(f"  - 召回率: {detected_errors/total_errors*100:.1f}%" if total_errors > 0 else "  - 召回率: N/A")
    print(f"  - 误报数: {false_positives}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
