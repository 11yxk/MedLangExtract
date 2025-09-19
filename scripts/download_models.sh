#!/bin/bash

echo "正在下载预训练模型..."

# 创建目录
mkdir -p lib/BiomedVLP-CXR-BERT-specialized
mkdir -p lib/convnext-tiny-224

echo "注意：请手动下载以下模型文件："
echo "1. CXR-BERT: https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized"
echo "   保存到: ./lib/BiomedVLP-CXR-BERT-specialized/"
echo ""
echo "2. ConvNeXt: https://huggingface.co/facebook/convnext-tiny-224" 
echo "   保存到: ./lib/convnext-tiny-224/"
echo ""
echo "或者修改config/training.yaml使用在线模型。"