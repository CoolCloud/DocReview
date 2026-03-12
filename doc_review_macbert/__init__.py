"""
基于 MacBERT 的公文校验模块

MacBERT 是针对中文优化的 BERT 模型，特别适合中文文本任务。
本模块用于检测公文中的常见错误，包括：
- 错别字
- 的地得混用
- 标点符号错误
- 同音字混淆
- 非正式用语等
"""

__version__ = "1.0.0"
__author__ = "DocReview Team"

from .model import MacBERTForDocReview
from .dataset import DocReviewDataset
from .predict import DocReviewPredictor

__all__ = [
    'MacBERTForDocReview',
    'DocReviewDataset',
    'DocReviewPredictor',
]
