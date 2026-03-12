"""
DistilBERT 情感分析模块
从 Colab notebook 提取并改进

使用示例:
    from sentiment_analysis import SentimentPredictor
    
    predictor = SentimentPredictor('./sentiment_models/best_model.pth')
    result = predictor.predict("Great restaurant!")
    print(result['sentiment'])
"""

from .model import CustomDistilBertForSequenceClassification, load_model, save_model
from .dataset import ReviewDataset, create_data_loaders
from .predict import SentimentPredictor, predict_sentiment

__version__ = "1.0.0"
__all__ = [
    'CustomDistilBertForSequenceClassification',
    'ReviewDataset',
    'SentimentPredictor',
    'create_data_loaders',
    'load_model',
    'save_model',
    'predict_sentiment'
]
