#!/bin/bash

# 公文核校系统 - 快速启动脚本

echo "========================================"
echo "公文核校系统 - 一键启动"
echo "========================================"

# 1. 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安装，请先安装 Python"
    exit 1
fi

echo "✓ Python 版本: $(python3 --version)"

# 2. 创建虚拟环境（可选）
if [ ! -d "venv" ]; then
    echo ""
    read -p "是否创建虚拟环境？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "创建虚拟环境..."
        python3 -m venv venv
        source venv/bin/activate
        echo "✓ 虚拟环境已激活"
    fi
else
    source venv/bin/activate
    echo "✓ 虚拟环境已激活"
fi

# 3. 安装依赖
echo ""
echo "安装依赖包..."
pip install -r requirements.txt

# 4. 生成训练数据
echo ""
echo "========================================"
echo "生成训练数据..."
echo "========================================"
python3 generate_data.py

# 5. 训练模型
echo ""
echo "========================================"
echo "开始训练模型..."
echo "========================================"
python3 train.py

# 6. 测试模型
echo ""
echo "========================================"
echo "测试模型..."
echo "========================================"
python3 inference.py

echo ""
echo "✓ 全部完成！"
