#!/usr/bin/env python3
"""
简单的公文核校演示脚本
"""
from inference import DocReviewer

def main():
    print("="*60)
    print("公文核校演示")
    print("="*60)
    print()
    
    # 加载模型
    reviewer = DocReviewer("models/best_model")
    
    # 演示样例
    demo_texts = [
        ("正常公文", "根据市政府的指示精神，现将有关事项通知如下。"),
        ("标点测试", "为了进一步加强管理。特制定本办法"),
        ("混合测试", "经本局研究决定，同意开展此项工作"),
    ]
    
    print("演示检测:\n")
    for title, text in demo_texts:
        print(f"【{title}】")
        print(f"文本: {text}")
        
        errors = reviewer.review(text)
        
        if errors:
            print(f"结果: ✗ 发现 {len(errors)} 处疑似错误")
            for err in errors:
                print(f"      - '{err['text']}'")
        else:
            print(f"结果: ✓ 未发现错误")
        print()

if __name__ == "__main__":
    main()
