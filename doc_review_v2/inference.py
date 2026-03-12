"""
公文核校推理脚本 V2
支持新训练的模型
"""
import torch
from transformers import BertTokenizer, BertForTokenClassification
import json
from pathlib import Path


# 标签映射
LABEL_MAP = {"O": 0, "B-ERROR": 1, "I-ERROR": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


class DocReviewer:
    """公文核校器 V2"""
    def __init__(self, model_path="models/best_model_v2"):
        print(f"加载模型: {model_path}")
        
        # 检测设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✓ 使用 Mac GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✓ 使用 CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("⚠ 使用 CPU")
        
        # 加载模型和 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ 模型加载完成\n")
    
    def review(self, text, return_details=False):
        """
        检查文本错误
        
        Args:
            text: 要检查的文本
            return_details: 是否返回详细信息（包含标签序列）
        
        Returns:
            errors: 错误列表
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=True
        )
        
        offset_mapping = encoding.pop('offset_mapping').squeeze().tolist()
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        
        # 如果只有一个token，转换为列表
        if isinstance(predictions, int):
            predictions = [predictions]
        
        # 解析错误位置
        errors = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        error_start = None
        error_tokens = []
        error_offsets = []
        
        for i, (pred, token, offset) in enumerate(zip(predictions, tokens, offset_mapping)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = ID2LABEL.get(pred, "O")
            
            if label == "B-ERROR":
                # 如果之前有未完成的错误，先保存
                if error_start is not None:
                    errors.append({
                        "start": error_offsets[0][0],
                        "end": error_offsets[-1][1],
                        "tokens": error_tokens,
                        "text": text[error_offsets[0][0]:error_offsets[-1][1]]
                    })
                # 开始新错误
                error_start = offset[0]
                error_tokens = [token]
                error_offsets = [offset]
            elif label == "I-ERROR" and error_start is not None:
                error_tokens.append(token)
                error_offsets.append(offset)
            else:
                # 错误结束
                if error_start is not None:
                    errors.append({
                        "start": error_offsets[0][0],
                        "end": error_offsets[-1][1],
                        "tokens": error_tokens,
                        "text": text[error_offsets[0][0]:error_offsets[-1][1]]
                    })
                    error_start = None
                    error_tokens = []
                    error_offsets = []
        
        # 处理最后一个错误
        if error_start is not None:
            errors.append({
                "start": error_offsets[0][0],
                "end": error_offsets[-1][1],
                "tokens": error_tokens,
                "text": text[error_offsets[0][0]:error_offsets[-1][1]]
            })
        
        if return_details:
            return {
                "errors": errors,
                "tokens": tokens,
                "labels": [ID2LABEL.get(p, "O") for p in predictions]
            }
        
        return errors


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def test_on_dataset():
    """在测试集上评估"""
    print("="*70)
    print("在测试集上评估模型")
    print("="*70 + "\n")
    
    reviewer = DocReviewer()
    
    # 加载测试数据
    test_data = load_jsonl("data/test.jsonl")
    
    # 只测试有错误的样本
    error_samples = [d for d in test_data if d['has_error']]
    
    print(f"测试集总数: {len(test_data)}")
    print(f"含错误样本: {len(error_samples)}")
    print("\n开始检测...\n")
    print("="*70)
    
    correct_count = 0
    total_errors_detected = 0
    total_errors_actual = 0
    
    for i, sample in enumerate(error_samples[:15], 1):  # 显示前15个
        text = sample['text']
        actual_errors = sample.get('errors', [])
        
        print(f"\n【样本 {i}】")
        print(f"文本: {text}")
        
        # 显示实际错误
        if actual_errors:
            print(f"\n实际错误 ({len(actual_errors)} 处):")
            for err in actual_errors:
                print(f"  - 位置 {err['position']}: '{err['correct_text']}' -> '{err['wrong_text']}'")
                print(f"    {err['error_desc']}")
        
        # 模型检测
        detected_errors = reviewer.review(text)
        
        total_errors_actual += len(actual_errors)
        total_errors_detected += len(detected_errors)
        
        if detected_errors:
            print(f"\n检测结果 ({len(detected_errors)} 处):")
            for err in detected_errors:
                print(f"  - 位置 {err['start']}-{err['end']}: '{err['text']}'")
        else:
            print("\n检测结果: ✓ 未发现错误")
        
        # 简单评估（检测到错误即认为正确）
        if len(detected_errors) > 0 and len(actual_errors) > 0:
            correct_count += 1
            print("✓ 检测正确")
        elif len(detected_errors) == 0 and len(actual_errors) == 0:
            correct_count += 1
            print("✓ 检测正确")
        else:
            print("✗ 检测有误")
        
        print("-" * 70)
    
    print(f"\n简单统计:")
    print(f"  正确检测: {correct_count}/15")
    print(f"  实际错误总数: {total_errors_actual}")
    print(f"  检测错误总数: {total_errors_detected}")


def interactive_mode():
    """交互式检测模式"""
    print("\n\n交互式检测模式（输入 'quit' 退出）:")
    print("="*70)
    
    reviewer = DocReviewer()
    
    while True:
        try:
            text = input("\n请输入要检测的公文内容: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            errors = reviewer.review(text)
            
            if errors:
                print(f"\n✗ 发现 {len(errors)} 处疑似错误:")
                for err in errors:
                    print(f"  - 位置 {err['start']}-{err['end']}: '{err['text']}'")
            else:
                print("\n✓ 未发现错误")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n感谢使用！")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # 测试模式
        test_on_dataset()
    else:
        # 交互模式
        print("="*70)
        print("公文核校系统 V2 - 推理测试")
        print("="*70 + "\n")
        
        reviewer = DocReviewer()
        
        # 快速演示
        demo_cases = [
            ("正常文本", "根据市政府的指示精神，现将有关事项通知如下。"),
            ("的地错误", "根据市政府地指示精神，现将有关事项通知如下。"),
            ("取得/取的", "此次培训取的了良好效果，达到了预期目标。"),
            ("贯切/贯彻", "现将实施方案印发给你们，请认真贯切执行。"),
        ]
        
        print("快速演示:\n")
        print("="*70)
        
        for title, text in demo_cases:
            print(f"\n【{title}】")
            print(f"文本: {text}")
            
            errors = reviewer.review(text)
            
            if errors:
                print(f"结果: ✗ 发现 {len(errors)} 处疑似错误")
                for err in errors:
                    print(f"      位置 {err['start']}-{err['end']}: '{err['text']}'")
            else:
                print(f"结果: ✓ 未发现错误")
        
        print("\n" + "="*70)
        
        # 进入交互模式
        interactive_mode()


if __name__ == "__main__":
    main()
