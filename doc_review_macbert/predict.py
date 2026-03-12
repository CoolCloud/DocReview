"""
MacBERT 公文校验预测模块
提供交互式预测和批量预测功能
"""
import torch
from transformers import BertTokenizer
from typing import List, Tuple, Dict
from pathlib import Path
import json

# 支持模块导入和直接运行
try:
    from .model import MacBERTForDocReview
    from .dataset import ID2LABEL
except ImportError:
    from model import MacBERTForDocReview
    from dataset import ID2LABEL


class DocReviewPredictor:
    """公文校验预测器"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化预测器
        
        参数:
            model_path: 模型目录路径
            device: 设备 ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_path = Path(model_path)
        
        # 选择设备
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"📥 从 {self.model_path} 加载模型...")
        print(f"🖥️  设备: {self.device}")
        
        # 加载 tokenizer 和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = MacBERTForDocReview.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ 模型加载完成\n")
    
    def predict(
        self,
        text: str,
        return_tokens: bool = True
    ) -> Dict:
        """
        对单个文本进行预测
        
        参数:
            text: 输入文本
            return_tokens: 是否返回 token 级别的预测
        
        返回:
            {
                'text': 原文本,
                'has_error': 是否有错误,
                'errors': [错误片段列表],
                'tokens': [token 预测列表] (可选)
            }
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=256
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'][0]
        
        # 预测
        with torch.no_grad():
            predictions = self.model.predict(input_ids, attention_mask)
        
        predictions = predictions[0].cpu().numpy()
        
        # 解析预测结果
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        errors = []
        current_error = None
        
        for i, (token, pred_id, (start, end)) in enumerate(zip(
            tokens, predictions, offset_mapping
        )):
            # 跳过特殊 token
            if start == 0 and end == 0:
                continue
            
            label = ID2LABEL.get(pred_id, "O")
            
            if label == "B-ERROR":
                # 新错误开始
                if current_error:
                    errors.append(current_error)
                current_error = {
                    'start': start,
                    'end': end,
                    'text': text[start:end],
                    'tokens': [token]
                }
            elif label == "I-ERROR" and current_error:
                # 错误延续
                current_error['end'] = end
                current_error['text'] = text[current_error['start']:end]
                current_error['tokens'].append(token)
            else:
                # 正确或错误结束
                if current_error:
                    errors.append(current_error)
                    current_error = None
        
        # 添加最后一个错误
        if current_error:
            errors.append(current_error)
        
        result = {
            'text': text,
            'has_error': len(errors) > 0,
            'errors': errors
        }
        
        if return_tokens:
            result['tokens'] = [
                {
                    'token': token,
                    'label': ID2LABEL.get(pred_id, "O"),
                    'position': (start, end)
                }
                for token, pred_id, (start, end) in zip(tokens, predictions, offset_mapping)
                if not (start == 0 and end == 0)  # 跳过特殊 token
            ]
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[Dict]:
        """
        批量预测
        
        参数:
            texts: 文本列表
            batch_size: 批次大小
        
        返回:
            预测结果列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = [self.predict(text, return_tokens=False) for text in batch_texts]
            results.extend(batch_results)
        
        return results
    
    def format_result(self, result: Dict, show_tokens: bool = False) -> str:
        """
        格式化预测结果为可读字符串
        
        参数:
            result: 预测结果字典
            show_tokens: 是否显示 token 级别的预测
        
        返回:
            格式化的字符串
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"📄 原文: {result['text']}")
        lines.append("=" * 60)
        
        if result['has_error']:
            lines.append(f"⚠️  检测到 {len(result['errors'])} 处错误:\n")
            for i, error in enumerate(result['errors'], 1):
                lines.append(f"  {i}. 位置 [{error['start']}:{error['end']}]")
                lines.append(f"     错误片段: 「{error['text']}」")
                lines.append("")
        else:
            lines.append("✓ 未检测到错误")
        
        if show_tokens and 'tokens' in result:
            lines.append("\n" + "-" * 60)
            lines.append("Token 级别预测:")
            lines.append("-" * 60)
            for token_info in result['tokens'][:50]:  # 只显示前 50 个
                label = token_info['label']
                if label != "O":
                    lines.append(
                        f"  {token_info['token']:10s} -> {label:10s} "
                        f"[{token_info['position'][0]}:{token_info['position'][1]}]"
                    )
        
        lines.append("=" * 60)
        return "\n".join(lines)


def interactive_mode(predictor: DocReviewPredictor):
    """交互式预测模式"""
    print("\n" + "=" * 60)
    print("🤖 MacBERT 公文校验 - 交互模式")
    print("=" * 60)
    print("输入公文文本进行校验，输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")
    
    while True:
        try:
            text = input("\n📝 请输入公文文本: ").strip()
            
            if not text:
                print("⚠️  输入为空，请重新输入")
                continue
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break
            
            # 预测
            result = predictor.predict(text, return_tokens=False)
            
            # 显示结果
            print("\n" + predictor.format_result(result))
        
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def demo_mode(predictor: DocReviewPredictor):
    """演示模式 - 使用预定义的测试样例"""
    print("\n" + "=" * 60)
    print("🎯 MacBERT 公文校验 - 演示模式")
    print("=" * 60 + "\n")
    
    # 测试样例
    test_cases = [
        {
            "text": "根据上级文件精神，认真组织学习。",
            "description": "正确样例"
        },
        {
            "text": "根据上级文件精神，任真组织学习。",
            "description": "错别字: 任真 -> 认真"
        },
        {
            "text": "为加强组织建设，经研究决定，坚定的推进工作。",
            "description": "的地得混用: 坚定的 -> 坚定地"
        },
        {
            "text": "会议讨论并通过了工作方案，取的了显著成效。",
            "description": "同音字错误: 取的 -> 取得"
        },
        {
            "text": "各单位要高度重视，贯切落实相关要求。",
            "description": "错别字: 贯切 -> 贯彻"
        },
        {
            "text": "关与此事的通知已经发布，请各部门认真执行。",
            "description": "同音字错误: 关与 -> 关于"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试样例 {i}: {case['description']}")
        print('='*60)
        
        result = predictor.predict(case['text'], return_tokens=False)
        print(predictor.format_result(result))
        
        if i < len(test_cases):
            input("\n按 Enter 继续...")


def batch_predict_file(
    predictor: DocReviewPredictor,
    input_file: str,
    output_file: str = None
):
    """
    从文件批量预测
    
    参数:
        predictor: 预测器
        input_file: 输入文件 (每行一个文本)
        output_file: 输出文件 (JSON 格式，可选)
    """
    print(f"\n📁 从文件加载数据: {input_file}")
    
    # 读取输入
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"✓ 加载 {len(texts)} 条数据\n")
    
    # 批量预测
    print("🔍 开始预测...")
    results = predictor.predict_batch(texts)
    
    # 统计
    error_count = sum(1 for r in results if r['has_error'])
    print(f"\n✓ 预测完成")
    print(f"  总样本数: {len(results)}")
    print(f"  含错误: {error_count} ({error_count/len(results)*100:.1f}%)")
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到: {output_file}")
    
    return results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MacBERT 公文校验预测")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models_macbert/best_model",
        help="模型路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['interactive', 'demo', 'batch'],
        default='demo',
        help="运行模式: interactive(交互), demo(演示), batch(批量)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="批量模式的输入文件"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="批量模式的输出文件"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help="设备选择"
    )
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = DocReviewPredictor(
        model_path=args.model_path,
        device=args.device
    )
    
    # 根据模式运行
    if args.mode == 'interactive':
        interactive_mode(predictor)
    elif args.mode == 'demo':
        demo_mode(predictor)
    elif args.mode == 'batch':
        if not args.input_file:
            print("❌ 批量模式需要指定 --input-file")
            return
        batch_predict_file(
            predictor,
            args.input_file,
            args.output_file
        )


if __name__ == "__main__":
    main()
