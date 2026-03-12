#!/bin/bash

# MacBERT 公文校验 - 快速启动脚本
# 自动完成数据生成、模型训练和演示的完整流程

echo "=================================="
echo "MacBERT 公文校验 - 快速启动"
echo "=================================="
echo ""

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ 未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

echo "✓ Python 版本: $(python --version)"
echo ""

# 步骤 1: 安装依赖
echo "步骤 1/4: 安装依赖..."
echo "--------------------------------"
pip install -q -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ 依赖安装完成"
else
    echo "❌ 依赖安装失败"
    exit 1
fi
echo ""

# 步骤 2: 生成训练数据
echo "步骤 2/4: 生成训练数据..."
echo "--------------------------------"
if [ ! -f "data/train.jsonl" ]; then
    python generate_data.py \
        --train 800 \
        --test 200 \
        --error-rate 0.4 \
        --output-dir data
    
    if [ $? -eq 0 ]; then
        echo "✓ 数据生成完成"
    else
        echo "❌ 数据生成失败"
        exit 1
    fi
else
    echo "✓ 数据已存在，跳过生成"
fi
echo ""

# 步骤 3: 训练模型
echo "步骤 3/4: 训练模型..."
echo "--------------------------------"
echo "⏰ 预计耗时: 10-30 分钟（取决于设备）"
echo ""

if [ ! -d "models/best_model" ]; then
    python train.py \
        --train-data data/train.jsonl \
        --test-data data/test.jsonl \
        --model-name hfl/chinese-macbert-base \
        --epochs 5 \
        --batch-size 16 \
        --output-dir models
    
    if [ $? -eq 0 ]; then
        echo "✓ 模型训练完成"
    else
        echo "❌ 模型训练失败"
        exit 1
    fi
else
    echo "✓ 模型已存在，跳过训练"
fi
echo ""

# 步骤 4: 运行演示
echo "步骤 4/4: 运行演示..."
echo "--------------------------------"
python predict.py \
    --model-path models/best_model \
    --mode demo

echo ""
echo "=================================="
echo "✅ 快速启动完成！"
echo "=================================="
echo ""
echo "📚 下一步："
echo "  1. 交互模式: python predict.py --mode interactive"
echo "  2. 批量预测: python predict.py --mode batch --input-file yourfile.txt"
echo "  3. Python API: 参考 README.md
echo ""
