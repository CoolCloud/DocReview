"""
MacBERT 模块基础测试
测试模块导入和基本功能（无需训练模型）
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("MacBERT 公文校验模块 - 基础测试")
print("=" * 60)
print()

# 测试 1: 导入模块
print("测试 1: 导入模块...")
try:
    import doc_review_macbert
    from doc_review_macbert.dataset import LABEL_MAP, ID2LABEL, NUM_LABELS
    from doc_review_macbert.model import MacBERTForDocReview
    print("✓ 模块导入成功")
    print(f"  - 标签映射: {LABEL_MAP}")
    print(f"  - 标签数量: {NUM_LABELS}")
except Exception as e:
    print(f"✗ 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 测试 2: 检查 transformers 和 torch
print("测试 2: 检查依赖...")
try:
    import torch
    import transformers
    print(f"✓ PyTorch 版本: {torch.__version__}")
    print(f"✓ Transformers 版本: {transformers.__version__}")
    
    # 检查设备
    if torch.backends.mps.is_available():
        print("✓ Mac GPU (MPS) 可用")
    elif torch.cuda.is_available():
        print(f"✓ CUDA GPU 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ 使用 CPU")
except Exception as e:
    print(f"✗ 依赖检查失败: {e}")
    sys.exit(1)

print()

# 测试 3: 创建模型（不加载预训练权重）
print("测试 3: 创建模型结构...")
try:
    from transformers import BertConfig
    from doc_review_macbert.model import MacBERTForDocReview
    
    config = BertConfig(
        vocab_size=21128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=NUM_LABELS
    )
    
    test_model = MacBERTForDocReview(config)
    total_params = sum(p.numel() for p in test_model.parameters())
    
    print(f"✓ 模型创建成功")
    print(f"  - 参数量: {total_params:,}")
    print(f"  - 标签数: {test_model.num_labels}")
except Exception as e:
    print(f"✗ 模型创建失败: {e}")
    sys.exit(1)

print()

# 测试 4: 测试数据生成脚本
print("测试 4: 测试数据生成功能...")
try:
    from doc_review_macbert.generate_data import ERROR_PATTERNS, generate_correct_sentence
    
    print(f"✓ 错误类型数量: {len(ERROR_PATTERNS)}")
    for error_type, info in ERROR_PATTERNS.items():
        print(f"  - {error_type}: {info['description']} ({len(info['examples'])} 个示例)")
    
    # 测试生成句子
    text, labels = generate_correct_sentence()
    print(f"✓ 生成测试句子: {text[:50]}...")
    print(f"  - 句子长度: {len(text)} 字")
    print(f"  - 标签数量: {len(labels)}")
except Exception as e:
    print(f"✗ 数据生成测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 测试 5: 测试 tokenizer（会下载预训练权重）
print("测试 5: 测试 tokenizer（可能需要下载）...")
try:
    from transformers import BertTokenizer
    
    print("  正在下载 MacBERT tokenizer（首次运行需要）...")
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
    
    test_text = "根据上级文件精神，认真组织学习。"
    encoding = tokenizer(test_text, return_tensors='pt')
    
    print(f"✓ Tokenizer 测试成功")
    print(f"  - 测试文本: {test_text}")
    print(f"  - Token 数量: {len(encoding['input_ids'][0])}")
    print(f"  - 词表大小: {len(tokenizer)}")
except Exception as e:
    print(f"⚠️  Tokenizer 测试失败（可能是网络问题）: {e}")
    print("   这不影响模块的其他功能")

print()

# 测试 6: 检查文件结构
print("测试 6: 检查模块文件...")
module_path = Path("doc_review_macbert")
required_files = [
    "__init__.py",
    "model.py",
    "dataset.py",
    "train.py",
    "predict.py",
    "generate_data.py",
    "requirements.txt",
    "README.md"
]

all_files_exist = True
for file in required_files:
    file_path = module_path / file
    if file_path.exists():
        print(f"✓ {file}")
    else:
        print(f"✗ {file} 缺失")
        all_files_exist = False

if not all_files_exist:
    print("⚠️  部分文件缺失")
else:
    print("✓ 所有文件完整")

print()

# 最终总结
print("=" * 60)
print("✅ 基础测试完成！")
print("=" * 60)
print()
print("📚 下一步:")
print("  1. 生成训练数据: python doc_review_macbert/generate_data.py")
print("  2. 训练模型: python doc_review_macbert/train.py")
print("  3. 运行演示: python doc_review_macbert/predict.py --mode demo")
print()
print("或者使用一键启动:")
print("  bash run_macbert.sh")
print()
