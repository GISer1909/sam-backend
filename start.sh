#!/bin/bash

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请安装 Python3"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "错误: 创建虚拟环境失败，请安装 python3-venv 包"
        exit 1
    fi
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 检查模型文件
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "警告: 未找到SAM模型文件 sam_vit_h_4b8939.pth"
    echo "请确保模型文件位于当前目录"
fi

# 创建必要的目录
mkdir -p uploads results

# 启动应用
echo "启动SAM标注系统..."
python app.py

# 如果应用退出，停用虚拟环境
deactivate
