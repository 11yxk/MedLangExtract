# 🚀 RobustMedSeg Quick Start Guide

> Robust medical image segmentation based on structured text extraction, the code will be updated progressively.

## 📋 Prerequisites

 **Google API**: Google API key required for LangExtract

## ⚡ Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/RobustMedSeg.git
cd RobustMedSeg

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install LangExtract
pip install langextract

# 4. Validate data
python examples/validate_data.py
```

## 🔑 Google API Configuration

```bash
# 1. Get API key
# Visit https://console.cloud.google.com/
# Enable Generative AI API
# Create API key

# 2. Configure extract.py
# Edit the api_key variable in extract.py
api_key = 'YOUR_GOOGLE_API_KEY'
```

## 📊 Data Processing Pipeline

### 1. Structured Text Extraction

```bash
# Use LangExtract to process original data
python extract.py

# This converts unstructured text to structured representation
# Input: "one medium yellow round polyp located in right of the image"
# Output: "[region_1] region_entity: polyp, size: medium, color: yellow, shape: round, location: right of the image"
```

### 2. Data Validation

```bash
# Validate all datasets
python examples/validate_data.py

# Output: 11 JSON files, 3 valid progressive learning files
```

## 🎯 Model Training

```bash
# Basic training (using structured data)
python train.py

# Custom configuration
python train.py --config config/training.yaml

# Monitor training process
tensorboard --logdir lightning_logs/
```

## 🔬 Robustness Evaluation

```bash
# Run comprehensive robustness testing
python examples/robustness_evaluation.py

# Test results include:
# - Original test set performance
# - Structured data performance (weak/strong perturbations)
# - Original perturbation data performance (comparison)
```


## 📁 Dataset Structure

```
data/
├── original_text_data/     # Original p0-p9 multi-level descriptions
│   ├── train.json         # Original training set
│   ├── val.json           # Original validation set
│   └── test.json          # Original test set
├── text_annotations/       # Structured annotation data
│   ├── kvasir_text/       # Progressive learning base data
│   ├── kvasir_aug/        # Structured augmented data (4 files)
│   └── kvasir_ori_aug/    # Original perturbation data (4 files)
└── Kvasir-SEG/            # Image data (requires download)
```

## 🛠️ Custom Usage

### Adapting to New Medical Domains

1. **Modify extraction rules**:
```python
# Modify in extract.py
custom_extraction_classes = [
    "lesion_type",      # Lesion type
    "severity",         # Severity level
    "texture",          # Texture features
    "anatomical_site"   # Anatomical site
]
```

2. **Update prompt description**:
```python
custom_prompt = """
Extract key information for [YOUR_DOMAIN] image analysis...
"""
```

### Processing Custom Data

```bash
# 1. Prepare original JSON data (containing detailed descriptions)
# 2. Configure extract.py paths and API key
# 3. Run structured extraction
python extract.py

# 4. Validate output quality
python examples/validate_data.py

# 5. Train robust model
python train.py
```

