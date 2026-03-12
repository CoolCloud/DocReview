"""
DistilBERT 情感分析预测脚本
从 Colab notebook 提取并改进
"""
import torch
from transformers import DistilBertTokenizerFast
import argparse

from model import CustomDistilBertForSequenceClassification


class SentimentPredictor:
    """情感分析预测器"""
    
    def __init__(self, model_path, tokenizer_path=None, device=None):
        """
        初始化预测器
        
        参数:
            model_path: 模型文件路径
            tokenizer_path: tokenizer 路径（如果为 None，使用 distilbert-base-uncased）
            device: 设备（如果为 None，自动选择）
        """
        # 设备
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # 加载 tokenizer
        if tokenizer_path is None:
            tokenizer_path = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
        
        # 加载模型
        self.model = CustomDistilBertForSequenceClassification(num_labels=3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 标签映射
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        print(f"✓ 模型已加载到 {self.device}")
    
    def predict(self, text, max_length=512):
        """
        预测单条文本的情感
        
        参数:
            text: 输入文本
            max_length: 最大序列长度
        
        返回:
            dict: {
                'text': 原文本,
                'sentiment': 预测情感标签,
                'confidence': 置信度,
                'probabilities': 所有类别的概率
            }
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # 所有类别的概率
        prob_dict = {
            self.id2label[i]: probabilities[0, i].item()
            for i in range(3)
        }
        
        return {
            'text': text,
            'sentiment': self.id2label[predicted_class],
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def predict_batch(self, texts, max_length=512, batch_size=16):
        """
        批量预测
        
        参数:
            texts: 文本列表
            max_length: 最大序列长度
            batch_size: batch 大小
        
        返回:
            结果列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # 预测
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
            
            # 收集结果
            for j, text in enumerate(batch_texts):
                pred_class = predicted_classes[j].item()
                confidence = probabilities[j, pred_class].item()
                
                prob_dict = {
                    self.id2label[k]: probabilities[j, k].item()
                    for k in range(3)
                }
                
                results.append({
                    'text': text,
                    'sentiment': self.id2label[pred_class],
                    'confidence': confidence,
                    'probabilities': prob_dict
                })
        
        return results


def predict_sentiment(review_text, model_path, tokenizer_path=None, max_length=512):
    """
    便捷函数：预测单条文本的情感
    （对应 Colab notebook 中的 predict_sentiment 函数）
    
    参数:
        review_text: 评论文本
        model_path: 模型路径
        tokenizer_path: tokenizer 路径
        max_length: 最大序列长度
    
    返回:
        情感标签字符串
    """
    predictor = SentimentPredictor(model_path, tokenizer_path)
    result = predictor.predict(review_text, max_length)
    return result['sentiment']


def main():
    parser = argparse.ArgumentParser(description='DistilBERT 情感分析预测')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Tokenizer 路径')
    parser.add_argument('--text', type=str, default=None, help='要预测的文本')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = SentimentPredictor(args.model_path, args.tokenizer_path)
    
    if args.interactive:
        # 交互模式
        print("\n" + "="*60)
        print("情感分析交互模式")
        print("输入文本进行情感分析，输入 'quit' 退出")
        print("="*60)
        
        while True:
            text = input("\n请输入评论文本: ").strip()
            
            if text.lower() == 'quit':
                print("退出...")
                break
            
            if not text:
                continue
            
            result = predictor.predict(text)
            
            print(f"\n结果:")
            print(f"  情感: {result['sentiment']}")
            print(f"  置信度: {result['confidence']:.4f}")
            print(f"  概率分布:")
            for label, prob in result['probabilities'].items():
                print(f"    {label}: {prob:.4f}")
    
    elif args.text:
        # 单次预测
        result = predictor.predict(args.text)
        print(f"\n文本: {result['text']}")
        print(f"情感: {result['sentiment']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"概率分布: {result['probabilities']}")
    
    else:
        print("请使用 --text 指定文本或 --interactive 进入交互模式")


if __name__ == "__main__":
    main()
