"""
公文核校推理脚本
加载训练好的模型进行错误检测
"""
import torch
from transformers import BertTokenizer, BertForTokenClassification
import json


# 标签映射
LABEL_MAP = {"O": 0, "B-ERROR": 1, "I-ERROR": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


class DocReviewer:
    """公文核校器"""
    def __init__(self, model_path="models/best_model"):
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
    
    def review(self, text):
        """检查文本错误"""
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
        
        # 解析错误位置
        errors = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        error_start = None
        error_tokens = []
        
        for i, (pred, token, offset) in enumerate(zip(predictions, tokens, offset_mapping)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = ID2LABEL.get(pred, "O")
            
            if label == "B-ERROR":
                if error_start is not None:
                    # 保存之前的错误
                    errors.append({
                        "start": error_start,
                        "tokens": error_tokens,
                        "text": "".join(error_tokens).replace("##", "")
                    })
                error_start = offset[0]
                error_tokens = [token]
            elif label == "I-ERROR" and error_start is not None:
                error_tokens.append(token)
            else:
                if error_start is not None:
                    errors.append({
                        "start": error_start,
                        "tokens": error_tokens,
                        "text": "".join(error_tokens).replace("##", "")
                    })
                    error_start = None
                    error_tokens = []
        
        # 处理最后一个错误
        if error_start is not None:
            errors.append({
                "start": error_start,
                "tokens": error_tokens,
                "text": "".join(error_tokens).replace("##", "")
            })
        
        return errors


def main():
    """测试推理"""
    print("="*60)
    print("公文核校系统 - 推理测试")
    print("="*60 + "\n")
    
    # 加载模型
    reviewer = DocReviewer()
    
    # 测试样例
    test_cases = [
        "根据上级部门的指示精神，现将有关事项通知如下。",  # 正常
        "根据上级部门地指示精神，现将有关事项通知如下。",  # 错误：地->的
        "为了进一步加强管理，特制订本办法。",  # 正常
        "经本局研究决定，同意开展此项工作，请遵照执行。",  # 正常
        "此次培训取的了良好效果，达到了预期目标。",  # 错误：取的->取得
    ]
    
    # 加载测试数据中的真实样例
    try:
        with open("data/test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
            # 添加一些有错误的测试样例
            for item in test_data[:5]:
                if item['has_error']:
                    test_cases.append(item['text'])
    except FileNotFoundError:
        print("未找到测试数据文件，使用默认测试样例\n")
    
    # 执行检测
    print("开始检测...\n")
    print("="*60)
    
    for i, text in enumerate(test_cases[:10], 1):
        print(f"\n【样例 {i}】")
        print(f"原文: {text}")
        
        errors = reviewer.review(text)
        
        if errors:
            print(f"✗ 发现 {len(errors)} 处疑似错误:")
            for err in errors:
                print(f"  - 位置 {err['start']}: '{err['text']}'")
        else:
            print("✓ 未发现错误")
        
        print("-" * 60)
    
    # 交互式模式
    print("\n\n交互式检测模式（输入 'quit' 退出）:")
    print("="*60)
    
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
                    print(f"  - '{err['text']}'")
            else:
                print("\n✓ 未发现错误")
        
        except KeyboardInterrupt:
            break
    
    print("\n感谢使用！")


if __name__ == "__main__":
    main()
