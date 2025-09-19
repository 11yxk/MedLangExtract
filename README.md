# 🚀 RobustMedSeg 快速开始指南

> 基于结构化文本提取的鲁棒医学图像分割

## 📋 前置条件

1. **Python环境**: Python 3.8+
2. **GPU支持**: 建议使用CUDA GPU进行训练
3. **存储空间**: 至少3GB可用空间
4. **Google API**: LangExtract需要Google API密钥

## ⚡ 快速安装

```bash
# 1. 克隆项目
git clone https://github.com/your-username/RobustMedSeg.git
cd RobustMedSeg

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装LangExtract
pip install langextract

# 4. 验证数据
python examples/validate_data.py
```

## 🔑 Google API配置

```bash
# 1. 获取API密钥
# 访问 https://console.cloud.google.com/
# 启用Generative AI API
# 创建API密钥

# 2. 配置extract.py
# 编辑extract.py中的api_key变量
api_key = 'YOUR_GOOGLE_API_KEY'
```

## 📊 数据处理流程

### 1. 结构化文本提取

```bash
# 使用LangExtract处理原始数据
python extract.py

# 这会将非结构化文本转换为结构化表示
# 输入: "one medium yellow round polyp located in right of the image"
# 输出: "[region_1] region_entity: polyp, size: medium, color: yellow, shape: round, location: right of the image"
```

### 2. 数据验证

```bash
# 验证所有数据集
python examples/validate_data.py

# 输出: 11个JSON文件，3个有效的progressive learning文件
```

## 🎯 模型训练

```bash
# 基础训练 (使用结构化数据)
python train.py

# 自定义配置
python train.py --config config/training.yaml

# 监控训练过程
tensorboard --logdir lightning_logs/
```

## 🔬 鲁棒性评估

```bash
# 运行完整鲁棒性测试
python examples/robustness_evaluation.py

# 测试结果包括:
# - 原始测试集性能
# - 结构化数据性能 (弱/强扰动)
# - 原始扰动数据性能 (对比)
```



## 📁 数据集结构

```
data/
├── original_text_data/     # 原始p0-p9多层级描述
│   ├── train.json         # 原始训练集
│   ├── val.json           # 原始验证集
│   └── test.json          # 原始测试集
├── text_annotations/       # 结构化标注数据
│   ├── kvasir_text/       # Progressive learning基础数据
│   ├── kvasir_aug/        # 结构化增强数据 (4文件)
│   └── kvasir_ori_aug/    # 原始扰动数据 (4文件)
└── Kvasir-SEG/            # 图像数据 (需下载)
```

## 🛠️ 自定义使用

### 适配新的医学领域

1. **修改提取规则**:
```python
# 在extract.py中修改
custom_extraction_classes = [
    "lesion_type",      # 病变类型
    "severity",         # 严重程度
    "texture",          # 纹理特征
    "anatomical_site"   # 解剖部位
]
```

2. **更新提示描述**:
```python
custom_prompt = """
Extract key information for [YOUR_DOMAIN] image analysis...
"""
```

### 处理自定义数据

```bash
# 1. 准备原始JSON数据 (包含详细描述)
# 2. 配置extract.py路径和API密钥
# 3. 运行结构化提取
python extract.py

# 4. 验证输出质量
python examples/validate_data.py

# 5. 训练鲁棒模型
python train.py
```

